import json
import argparse
import random
import itertools
import re
from typing import Dict, List, Any, Tuple

from utils.assertions import AUTHORITY_SRCS, BELIEF_SRCS
from utils.io import read_jsonl_with_jsonlines

class AssertionGenerator:
    def __init__(self, template_file: str):
        """Initialize with a JSON template file."""
        with open(template_file, 'r') as f:
            self.templates = json.load(f)
        
        # Default placeholder values for testing
        self.placeholders = {
            "subject": "the capital of France",
            "object": "London",
            "subject_relation": "capital",
            "object_relation": "capital",
            "extra_information": "hosted the Olympics in 2016",
            "condition": "Berlin is the capital of Germany",
            "consequence": "tourism patterns would be different",
            "object_property": "contains Buckingham Palace",
            "implied_object_property": "Buckingham Palace",
            "authority_source": "Wikipedia",
            "belief_source": "My professor",
            "formal_subject": "the French Republic",
            "formal_subject_relation": "sovereign capital",
            "formal_object": "London",
            "poetic_subject_descriptor": "storied capitals of Europe",
            "subject_descriptor": "French king"
        }
    
    def get_random_template(self, dimension: str, category: str) -> str:
        """Get a random template from a specific dimension and category."""
        templates = self.templates[dimension][category]["templates"]
        return random.choice(templates)
    
    def fill_template(self, template: str, placeholders: Dict[str, str] = None) -> str:
        """Fill a template with placeholders."""
        if placeholders is None:
            placeholders = self.placeholders
        
        result = template
        for key, value in placeholders.items():
            placeholder = "{" + key + "}"
            if placeholder in result:
                result = result.replace(placeholder, value)
        
        return result[0].upper() + result[1:]
    
    def generate_assertion(self, dimension: str, category: str, placeholders: Dict[str, str] = None) -> str:
        """Generate an assertion for a specific dimension and category."""
        template = self.get_random_template(dimension, category)
        assertion = self.fill_template(template, placeholders)
        
        def has_templates(text):
            return bool(re.search(r'\{[^}]+\}', text))
        
        if has_templates(assertion):
            return None
        
        return assertion

    def generate_queries(self, fact: Dict[str, str]) -> str:
        """Generate a query for a specific fact."""
        return f'Is {fact["object_pri"]} {fact["subject_relation"]}?', f'Is {fact["object_ctx"]} {fact["subject_relation"]}?'
    
    def generate_cross_dimensional_assertion(self, 
                                           dimensions: List[Tuple[str, str]], 
                                           placeholders: Dict[str, str] = None) -> str:
        """
        Generate an assertion that combines multiple dimensions.
        
        Args:
            dimensions: List of (dimension, category) tuples in order of application
            placeholders: Dictionary of placeholder values
            
        Returns:
            A string with the generated assertion
        """
        if not dimensions:
            return ""
        
        # Use default placeholders if none provided
        if placeholders is None:
            placeholders = self.placeholders
        
        # Start with the first dimension as the base
        base_dim, base_cat = dimensions[0]
        assertion = self.generate_assertion(base_dim, base_cat, placeholders)
        
        # For each additional dimension, modify the assertion
        for dimension, category in dimensions[1:]:
            # Special cases for combining dimensions
            if dimension == "evidentiality":
                if category == "authority":
                    authority = placeholders.get("authority_source", "Wikipedia")
                    assertion = f"According to {authority}, {assertion}"
                elif category == "belief_reports":
                    belief_source = placeholders.get("belief_source", "My professor")
                    assertion = f"{belief_source} believes that {assertion}"
                elif category == "hearsay":
                    assertion = f"I've heard that {assertion}"
            
            elif dimension == "epistemic_stance":
                if category == "strong":
                    assertion = f"It is certain that {assertion}"
                elif category == "weak":
                    assertion = f"Perhaps {assertion}"
            
            # For tone, we might need more complex rules
            elif dimension == "tone":
                # This is simplified - in a real implementation you'd need more sophisticated rules
                if base_dim == "form" and base_cat == "explicit":
                    # We can directly apply tone templates
                    tone_template = self.get_random_template(dimension, category)
                    assertion = self.fill_template(tone_template, placeholders)
            
        return assertion
    
    def generate_dataset(self, facts: List[Dict[str, str]], dimensions_to_vary: List[str]) -> List[Dict[str, Any]]:
        """
        Generate a dataset by systematically varying dimensions for each fact.
        
        Args:
            facts: List of dictionaries with placeholder values for each fact
            dimensions_to_vary: List of dimensions to systematically vary
            
        Returns:
            List of dictionaries with generated assertions
        """
        dataset = []
        
        for fact in facts:
            fact_examples = []
            
            # For each dimension to vary
            for dimension in dimensions_to_vary:
                # For each category in this dimension
                for category in self.templates[dimension].keys():
                    # Generate an assertion with this dimension/category
                    assertion = self.generate_assertion(dimension, category, fact)
                    pri_query, ctx_query = self.generate_queries(fact)
                    if assertion is not None:
                        fact_examples.append({
                            "fact": fact,
                            "dimension": dimension,
                            "category": category,
                            "assertion": assertion,
                            "query": pri_query,
                            "query_type": "prior_yes",
                        })
                        fact_examples.append({
                            "fact": fact,
                            "dimension": dimension,
                            "category": category,
                            "assertion": assertion,
                            "query": ctx_query,
                            "query_type": "ctx_yes",
                        })
            
            dataset.extend(fact_examples)
            
        return dataset
    
    def generate_balanced_dataset(self, facts: List[Dict[str, str]], 
                            dimension_categories: Dict[str, List[str]], 
                            num_facts: int = 50) -> List[Dict[str, Any]]:
        """
        Generate a balanced dataset with equal samples per dimension-category combination.
        
        Args:
            facts: List of fact dictionaries
            dimension_categories: Dict mapping dimensions to their categories
            num_facts: Number of facts to sample for generating combinations
            
        Returns:
            List of balanced assertion examples
        """
        balanced_dataset = []
        failed_generations = []
        
        total_combinations = sum(len(cats) for cats in dimension_categories.values())
        print(f"Generating balanced dataset: {num_facts} facts × {total_combinations} combinations")
        
        # Sample facts for this run
        sampled_facts = random.sample(facts, min(num_facts, len(facts)))
        
        
        for i, fact in enumerate(sampled_facts):
            for dimension, categories in dimension_categories.items():
                combination_examples = []
                for category in categories:
                    print(f"Generating {dimension}-{category}...")
                    try:
                        # Create placeholders from fact
                        placeholders = self.fact_to_placeholders(fact)
                        
                        # Generate assertion
                        assertion = self.generate_assertion(dimension, category, placeholders)                  
                        
                        if assertion is not None:
                            pri_query, ctx_query = self.generate_queries(fact)
                            
                            combination_examples.append({
                                "fact": fact,
                                "dimension": dimension,
                                "category": category,
                                "assertion": assertion,
                                "query": pri_query,
                                "query_type": "prior_yes",
                                "placeholders": placeholders
                            })
                            combination_examples.append({
                                "fact": fact,
                                "dimension": dimension,
                                "category": category,
                                "assertion": assertion,
                                "query": ctx_query,
                                "query_type": "ctx_yes",
                                "placeholders": placeholders
                            })
                        else:
                            print(f"DEBUG: Template unfilled for {dimension}-{category}")
                            print(f"Available placeholders: {list(placeholders.keys())}")
                            print(f"Fact: {fact}")
                            failed_generations.append({
                                "dimension": dimension,
                                "category": category,
                                "fact_index": i,
                                "reason": "unfilled_template"
                            })
                            
                    except Exception as e:
                        failed_generations.append({
                            "dimension": dimension,
                            "category": category,
                            "fact_index": i,
                            "reason": str(e)
                        })
                
                balanced_dataset.extend(combination_examples)
                print(f"  Generated {len(combination_examples)} assertions")
        
        print(f"\nBalance generation complete:")
        print(f"  Total assertions: {len(balanced_dataset)}")
        print(f"  Failed generations: {len(failed_generations)}")
        
        if failed_generations:
            print("  Failed generation breakdown:")
            failure_counts = {}
            for failure in failed_generations:
                key = f"{failure['dimension']}-{failure['category']}"
                failure_counts[key] = failure_counts.get(key, 0) + 1
            for key, count in failure_counts.items():
                print(f"    {key}: {count} failures")
        
        return balanced_dataset, failed_generations
    
    def fact_to_placeholders(self, fact: Dict[str, str]) -> Dict[str, str]:
        """Convert a fact dictionary to template placeholders."""
        placeholders = {
            "subject": fact.get("subject", ""),
            "object_ctx": fact.get("object_ctx", ""),
            "object_pri": fact.get("object_pri", ""),
            "subject_relation": fact.get("subject_relation", ""),
            "relation": fact.get("relation", ""),
            "extra_info_obj_ctx": fact.get("extra_info_obj_ctx", ""),
            "condition": fact.get("condition", ""),
            "counterfactual_condition": fact.get("counterfactual_condition", ""),
            "authority_source": fact.get("authority_source", "Wikipedia"),
            "belief_source": fact.get("belief_source", "my professor"),
            # Add any other mappings the templates need
        }
        
        # Add computed placeholders
        if "object_ctx" in fact and "subject_relation" in fact:
            placeholders["object"] = fact["object_ctx"]  # For backwards compatibility
        
        return placeholders
    
    def generate_cross_dimensional_dataset(self, facts: List[Dict[str, str]], 
                                        dimension_pairs: List[List[Tuple[str, str]]]) -> List[Dict[str, Any]]:
        """
        Generate a dataset with cross-dimensional variations.
        
        Args:
            facts: List of dictionaries with placeholder values for each fact
            dimension_pairs: List of lists of (dimension, category) tuples to combine
            
        Returns:
            List of dictionaries with generated assertions
        """
        dataset = []
        
        for fact in facts:
            for dimensions in dimension_pairs:
                assertion = self.generate_cross_dimensional_assertion(dimensions, fact)
                
                dataset.append({
                    "fact": fact,
                    "dimensions": dimensions,
                    "assertion": assertion
                })
                
        return dataset

def load_and_preprocess_yago_sample(file_path: str) -> List[Dict[str, str]]:
    """
    Load the YAGO sample and preprocess it to create a list of facts. 
    Each fact should contain the subject, object, subject_relation, object_relation=None, relation_name, category_name.
    For example,
    {
        "subject": "the author of Harry Potter",
        "object": "JK Rowling",
        "subject_relation": "author",
        "object_relation": None,
        "relation_name": "http://schema.org/author",
        "category_name": "singlelabel_stable"
    }
    """
    with open(file_path, "r") as f:
        yago_sample = json.load(f)
    facts = []
    for cat, rels_dict in yago_sample.items():
        for rel, rel_dict in rels_dict.items():
            for entity, answer_wrong, answer in zip(rel_dict["entities"], rel_dict["answers_wrong"], rel_dict["answers"]):
                facts.append({
                    "subject": rel_dict["subject"].format(entity=entity),
                    "object": rel_dict["object"].format(answer=answer_wrong),
                    "object_true": rel_dict["object"].format(answer=answer),
                    "subject_relation": rel_dict["subject_relation"],
                    "object_relation": None,
                    "relation_name": rel,
                    "category_name": cat
                })
    return facts

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate balanced assertion datasets")
    parser.add_argument(
        "-N", "--num-facts",
        type=int,
        default=1000,
        help="Number of facts to sample for generation",
    )
    args = parser.parse_args()

    generator = AssertionGenerator("preprocessing/assertion_templates.json")
    
    # Generate a simple assertion
    print("SINGLE DIMENSION EXAMPLES:")
    print("Form (explicit):", generator.generate_assertion("form", "explicit"))
    print("Form (not at-issue):", generator.generate_assertion("form", "not_at_issue"))
    print("Directness (indirect_entailment):", generator.generate_assertion("directness", "indirect_entailment"))
    print("Evidentiality (authority):", generator.generate_assertion("evidentiality", "authority"))
    print("Epistemic stance (weak):", generator.generate_assertion("epistemic_stance", "weak"))
    print("Tone (formal):", generator.generate_assertion("tone", "formal"))
    
    print("\nCROSS-DIMENSIONAL EXAMPLES:")
    # Generate cross-dimensional assertions
    cross_dims = [
        [("form", "not_at_issue"), ("evidentiality", "authority")],
        [("form", "counterfactual"), ("epistemic_stance", "weak")],
        [("form", "explicit"), ("tone", "formal")],
        [("form", "interrogative"), ("evidentiality", "hearsay")]
    ]
    
    for dims in cross_dims:
        dim_str = " + ".join([f"{d[0]} ({d[1]})" for d in dims])
        print(f"{dim_str}:", generator.generate_cross_dimensional_assertion(dims))
    
    print("\nDATASET GENERATION EXAMPLE:")
    # Generate a dataset with different facts
    facts = read_jsonl_with_jsonlines("data/popqa_filtered_v2_enhanced.jsonl")  # Change fact dataset here that should be used 
    aug_facts = []
    for fact in facts:
        for authority_src, belief_src in zip(AUTHORITY_SRCS, BELIEF_SRCS):
            aug_fact = fact.copy()
            aug_fact["authority_source"] = authority_src
            aug_fact["belief_source"] = belief_src
            aug_facts.append(aug_fact)
    
    # Generate examples varying the form dimension
    # old version that creates for every fact all 17 combinations
    # dataset = generator.generate_dataset(facts, ["form", "epistemic_stance", "evidentiality", "tone"])

    # new version that creates exactly samples_per_combination for all 17 combinations
    dimension_categories = {
        "form": ["explicit", "conditional", "counterfactual", "imperative", "interrogative", "not_at_issue"],
        "epistemic_stance": ["strong", "weak"],
        "evidentiality": ["hearsay", "authority", "belief_reports"],
        "tone": ["informal", "poetic", "child_directed", "emotional_appeal", "sarcasm", "social_media"]
    }

    dataset, failed_generations = generator.generate_balanced_dataset(
        facts=facts, 
        dimension_categories=dimension_categories,
        num_facts=args.num_facts # Number of facts sampled for generation
    )
    if len(failed_generations)>0:
        print(f"There were {len(failed_generations)} failed generations!")
    print(len(dataset))

    with open(f"data/generated_assertions_v2_{args.num_facts}.jsonl", "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")
    
    # Print a few examples
    for i, example in enumerate(dataset[:4]):
        print(f"Example {i+1}: {example['assertion']}")
        print(f"  Dimension: {example['dimension']}, Category: {example['category']}")
        print(f"  Fact: {example['fact']['subject_relation']} - {example['fact']['object_ctx']}")
        print(f"  Query: {example['query']}")