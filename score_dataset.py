import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import os
from pathlib import Path
import argparse
from tqdm import tqdm
import torch.nn.functional as F
import logging
from utils.model import load_model_and_tokenizer, to_chat_template
from utils.assertions import AUTHORITY_SRCS, BELIEF_SRCS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_yes_no_probabilities(model, tokenizer, prompt):
    try:
        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Get model outputs
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last token logits
        
        # Get tokens for "Yes" and "No" - try with space prefix first
        yes_tokens = tokenizer.encode(" Yes", add_special_tokens=False)
        no_tokens = tokenizer.encode(" No", add_special_tokens=False)
        
        # If the space-prefixed versions are empty, try without space
        if not yes_tokens:
            yes_tokens = tokenizer.encode("Yes", add_special_tokens=False)
        if not no_tokens:
            no_tokens = tokenizer.encode("No", add_special_tokens=False)
        
        # Use the first token of each
        yes_token_id = yes_tokens[0]
        no_token_id = no_tokens[0]
        
        # Convert logits to probabilities over the full vocabulary
        full_probabilities = F.softmax(logits, dim=0)
        
        # Extract probabilities for "Yes" and "No" tokens
        yes_prob = full_probabilities[yes_token_id].item()
        no_prob = full_probabilities[no_token_id].item()
        
        return yes_prob, no_prob
    
    except Exception as e:
        logger.error(f"Error getting probabilities for prompt: {prompt[:100]}... Error: {e}")
        return 0.5, 0.5  # Return neutral probabilities on error

def generate_answer(model, tokenizer, prompt, max_length=20):
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=False,  # Greedy decoding
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                temperature=None,  
                top_p=None 
            )
        
        # Decode only the new tokens
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        return answer
    
    except Exception as e:
        logger.error(f"Error generating answer for prompt: {prompt[:100]}... Error: {e}")
        return "ERROR"

def classify_answer(answer):
    """
    Classify the answer as:
    - 'memory': agreeing with memorized/true information (answering Yes to queries like "Is Paris the capital of France?")
    - 'context': agreeing with the assertion/context (answering No to the same query when context says "London is capital of France")
    - 'other': anything else
    """
    if answer == "ERROR":
        return 'error'
    
    answer_lower = answer.lower().strip()
    
    if answer_lower.startswith('yes'):
        return 'memory'  # Agreeing with true facts
    elif answer_lower.startswith('no'):
        return 'context'  # Agreeing with assertion
    else:
        return 'other'

def process_dataset(input_file, model_name, output_dir):
    # Load model and tokenizer
    logger.info(f"Loading model: {model_name}")
    model, tokenizer = load_model_and_tokenizer(model_name, load_in_4bit=False, load_in_8bit=False, train_mode=False, dtype=torch.bfloat16, device="auto")
    
    # Load dataset
    logger.info(f"Loading dataset from {input_file}")
    examples = []
    try:
        with open(input_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_file}")
        raise
    
    logger.info(f"Loaded {len(examples)} examples")
    
    results = []
    
    logger.info("Processing examples...")
    for i, example in enumerate(tqdm(examples)):
        try:
            assertion = example['assertion']
            query = example['query']
            prompt = to_chat_template(f"{assertion} {query}", tokenizer)
            
            # Get generated answer
            answer = generate_answer(model, tokenizer, prompt, max_length=5)
            
            # Get yes/no probabilities
            yes_prob, no_prob = get_yes_no_probabilities(model, tokenizer, prompt)
            
            # Classify the answer
            classification = classify_answer(answer)
            
            # Store results
            result = {
                'example_id': i,
                'assertion': assertion,
                'query': query,
                'prompt': prompt,
                'generated_answer': answer,
                'yes_probability': yes_prob,
                'no_probability': no_prob,
                'classification': classification,
                'dimension': example.get('dimension', ''),
                'category': example.get('category', ''),
                'subject': example['fact']['subject'],
                'object': example['fact']['object_ctx'],
                'object_true': example['fact']['object_pri']
            }
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing example {i}: {e}")
            # Add a placeholder result for this example
            result = {
                'example_id': i,
                'assertion': example.get('assertion', ''),
                'query': example.get('query', ''),
                'prompt': '',
                'generated_answer': 'ERROR',
                'yes_probability': 0.5,
                'no_probability': 0.5,
                'classification': 'error',
                'dimension': example.get('dimension', ''),
                'category': example.get('category', ''),
                'subject': example.get('fact', {}).get('subject', ''),
                'object': example.get('fact', {}).get('object_ctx', ''),
                'object_true': example.get('fact', {}).get('object_pri', ''),
                "condition": example.get('condition', ''),
                "extra_information": example.get('extra_information', ''),
                "counterfactual_condition": example.get('counterfactual_condition', ''),
                "authority_source": example.get('authority_source', ''),
                "belief_source": example.get('belief_source', ''),
            }
            results.append(result)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save results
    df = pd.DataFrame(results)
    output_file = os.path.join(output_dir, 'results.csv')
    df.to_csv(output_file, index=False)
    logger.info(f"Results saved to: {output_file}")
    
    # Calculate statistics
    total = len(results)
    memory_count = sum(1 for r in results if r['classification'] == 'memory')
    context_count = sum(1 for r in results if r['classification'] == 'context')
    other_count = sum(1 for r in results if r['classification'] == 'other')
    error_count = sum(1 for r in results if r['classification'] == 'error')
    
    memory_pct = (memory_count / total) * 100 if total > 0 else 0
    context_pct = (context_count / total) * 100 if total > 0 else 0
    other_pct = (other_count / total) * 100 if total > 0 else 0
    error_pct = (error_count / total) * 100 if total > 0 else 0
    
    logger.info(f"\nStatistics:")
    logger.info(f"Memory-agreeing answers: {memory_count}/{total} ({memory_pct:.1f}%)")
    logger.info(f"Context-agreeing answers: {context_count}/{total} ({context_pct:.1f}%)")
    logger.info(f"Other answers: {other_count}/{total} ({other_pct:.1f}%)")
    logger.info(f"Error answers: {error_count}/{total} ({error_pct:.1f}%)")
    
    # Save summary statistics
    summary = {
        'model_name': model_name,
        'total_examples': total,
        'memory_agreeing_count': memory_count,
        'context_agreeing_count': context_count,
        'other_count': other_count,
        'error_count': error_count,
        'memory_agreeing_pct': memory_pct,
        'context_agreeing_pct': context_pct,
        'other_pct': other_pct,
        'error_pct': error_pct,
        'avg_yes_probability': df['yes_probability'].mean(),
        'avg_no_probability': df['no_probability'].mean()
    }
    
    summary_file = os.path.join(output_dir, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Summary saved to: {summary_file}")
    
    return summary

def main():
    parser = argparse.ArgumentParser(description="Score assertion dataset using HuggingFace models")
    parser.add_argument('--input_file', default='data/generated_assertions_v2_500.jsonl',  
                   help='Path to input JSONL file')
    parser.add_argument('--model_name', default='meta-llama/Llama-3.1-8B-Instruct',
                       help='HuggingFace model name')
    parser.add_argument('--output_dir', default=None,
                       help='Output directory (default: data/{model_name_safe}_{dataset_name})')
    
    args = parser.parse_args()
    
    # Create safe model name for directory
    model_name_safe = args.model_name.replace('/', '_').replace('-', '_').replace('.', '_')

    input_path = Path(args.input_file)
    dataset_name = input_path.stem 
    output_dir = args.output_dir or f"data/{model_name_safe}_{dataset_name}"
    
    try:
        summary = process_dataset(args.input_file, args.model_name, output_dir)
        logger.info("Processing completed successfully!")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise

if __name__ == "__main__":
    main() 