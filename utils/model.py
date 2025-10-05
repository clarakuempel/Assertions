import os
import gc
import hashlib
import subprocess as sp
from typing import Optional, List, Union, Dict, Tuple
from enum import Enum
from tqdm import tqdm

import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from peft import AutoPeftModelForCausalLM, prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftConfig
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments

# Use DLAB model hub
try:
    from dlabutils import model_path
    print("Using DLAB model hub.")
except:
    model_path = lambda path: path

class TokenType(Enum):
    ProName = "pro_name"
    Period = "period"
    ProNameQuestion = "pro_name_question"
    QuestionMark = "question_mark"

#################
# MODEL LOADING #
#################
def load_model_and_tokenizer(
    model_id: str,
    load_in_4bit: bool,
    load_in_8bit: bool,
    train_mode: bool = True,
    peft_config: Optional[PeftConfig] = None,
    dtype: Optional[str] = "auto",
    device: str = "auto",
    attn_implementation: str = "sdpa",
    try_load_as_peft: bool = False,
    padding_side: str = "right",
):
    """
    Load the model and tokenizer from huggingface.
    Args:
        model_id: str
        load_in_4bit: bool -  whether to use 4bit quantization to reduce memory usage.
            # 4bit pre quantized models we support for 4x faster downloading + no OOMs.
            fourbit_models = [
                "unsloth/mistral-7b-bnb-4bit",
                "unsloth/mistral-7b-v0.2-bnb-4bit", # New Mistral 32K base model
                "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
                "unsloth/llama-2-7b-bnb-4bit",
                "unsloth/llama-2-13b-bnb-4bit",
                "unsloth/codellama-34b-bnb-4bit",
                "unsloth/tinyllama-bnb-4bit",
                "unsloth/gemma-7b-bnb-4bit", # New Google 6 trillion tokens model 2.5x faster!
                "unsloth/gemma-2b-bnb-4bit",
            ] # More models at https://huggingface.co/unsloth
        load_in_8bit: bool
        dtype: torch.dtype - default to None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        device: str - default to auto
    """
    model_id = model_path(model_id)
    if "Meta-Llama-3.2" in model_id:
        model_id = model_id.replace("Meta-Llama-3.2-", "Llama-3.2-")
    print(f"Loading model '{model_id}'")
    if load_in_4bit and load_in_8bit:
        raise ValueError("Cannot load in both 4bit and 8bit.")

    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif load_in_8bit:
        # TODO(kdu): untested
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        bnb_config = None

    if peft_config is not None or try_load_as_peft:
        try:
            model = AutoPeftModelForCausalLM.from_pretrained(
                model_id,
                is_trainable=train_mode,
                config=peft_config,
                quantization_config=bnb_config,
                device_map=device,
                torch_dtype=dtype,
                attn_implementation=attn_implementation,
            )
            tokenizer = prepare_tokenizer(model)
        except ValueError:
            print("Failed to load model with AutoPeftModelForCausalLM, now attempting with AutoModelForCausalLM.")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map=device,
                torch_dtype=dtype,
                attn_implementation=attn_implementation,
            )
            tokenizer = prepare_tokenizer(model, padding_side=padding_side)
            if train_mode:
                # If we are not training the model, we do not want to load it in peft mode
                model = prepare_peft_model(model, peft_config=peft_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=device,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
        )
        tokenizer = prepare_tokenizer(model, padding_side=padding_side)
    print(f"Loaded model on device {model.device} with dtype {model.dtype}.")

    torch.cuda.empty_cache()
    gc.collect()

    return model, tokenizer


def prepare_tokenizer(model, set_pad_token=True, padding_side="right"):
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)

    tokenizer.padding_side = padding_side  # for kbit training apparently you need to pad on the right
    if set_pad_token and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def prepare_peft_model(
    model, peft_config, target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj"], **lora_config_kwargs
):
    """
    Args:
        target modules - subset of ["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2", "gate_proj", "up_proj", "down_proj"]
    """
    model.gradient_checkpointing_disable()
    model = prepare_model_for_kbit_training(model)  # model becomes float32 instead of bfloat16
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def merge_save_peft(peft_model, tokenizer, path):
    """Merge the peft model and save to path."""

    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    tokenizer.padding_side = "left"

    return merged_model, tokenizer


##############
# EVALUATION #
##############
def response_startswith_label(response: str, label: str) -> bool:
    return response.startswith(label)


def compute_mr(df, is_response_correct_func=response_startswith_label) -> Tuple[float, float, float]:
    """
    Given a df with columns `predictions`,  `prior_answer`, and `ctx_answer`, return a tuple containing (MR, % of other answers).
    MR = (# prior) / (# prior + # context)
    CR = (# ctx) / (# prior + # context)
    % other answers = (# other) / (# prior + # context + # other)
    """
    if len(df) == 0:
        return None, None
    num_prior_answers = df.apply(
        lambda row: is_response_correct_func(row["predictions"], row["prior_answer"]), axis=1
    ).sum()
    num_ctx_answers = df.apply(
        lambda row: is_response_correct_func(row["predictions"], row["ctx_answer"]), axis=1
    ).sum()
    num_other_answers = len(df) - (num_ctx_answers + num_prior_answers)
    if num_prior_answers + num_ctx_answers == 0:
        print("No correct prior or context answers. Returning None")
        return None, num_other_answers / len(df)
    return num_prior_answers / (num_prior_answers + num_ctx_answers), num_other_answers / len(df)


def compute_pair_acc(df):
    return df.groupby(["context", "query"]).agg("min")["is_correct"].mean()


def compute_metrics(df, is_response_correct_func=response_startswith_label):
    ctx_pref_df = df[df["weight_context"] == 1.0]
    prior_pref_df = df[df["weight_context"] == 0.0]

    context_acc = ctx_pref_df["is_correct"].mean()
    prior_acc = prior_pref_df["is_correct"].mean()

    context_mr, context_other = compute_mr(ctx_pref_df, is_response_correct_func=is_response_correct_func)
    prior_mr, prior_other = compute_mr(prior_pref_df, is_response_correct_func=is_response_correct_func)

    overall_mr, overall_other = compute_mr(df)

    pair_acc = compute_pair_acc(df)

    metrics = {
        "acc": df["is_correct"].mean(),  # Overall accuracy
        "pair_acc": pair_acc,
        "context_acc": context_acc,  # accuracy across the examples that SHOULD follow the context
        "prior_acc": prior_acc,  # accuracy across the examples that SHOULD follow the prior
        "context_mr": context_mr,  # MR across the examples that SHOULD follow the context (we want this to be low)
        "prior_mr": prior_mr,  # MR across the examples that SHOULD follow the context (we want this to be high)
        "overall_mr": overall_mr,  # MR across all examples (we want this to be 50%)
        "context_pct_other": context_other,  # percent of examples featured a non-context or prior answer across examples that SHOULD follow the context (lower better)
        "prior_pct_other": prior_other,  # percent of examples that featured a non-context or prior answer across examples that SHOULD follow the prior (lower better)
        "overall_pct_other": overall_other,  # percent of examples that featured a non-context or prior answer across all examples (lower better)
    }

    if "query_only_is_correct" in df.columns:
        metrics["query_only_acc"] = df["query_only_is_correct"].mean()

    return metrics


def compute_metrics_only_og_correct(df):
    metrics = compute_metrics(df[df["query_only_is_correct"] == True])  # noqa
    del metrics["query_only_acc"]
    return metrics


def evaluate_model_queries_only(
    model,
    tokenizer,
    dataset: Dataset,
    max_new_tokens: int = 30,
    batch_sz: int = 8,  # "auto",
    device: str = "auto",
    is_response_correct_func=response_startswith_label,
    hook=None,
):
    """
    Given a dataset with columns ["query", "prior_answer"], generate answers and evaluate model accuracy against those labels.
    1. Generate predictions from text
    2. Extract answer, compare to labels, and return accuracy
    """
    # Free gpu memory
    gc.collect()
    torch.cuda.empty_cache()
    if batch_sz == "auto":
        batch_sz = int(2 * int(sum(get_gpu_memory()) / 1000))
        print(f"Setting batch size to {batch_sz} for eval.")

    tokenizer.padding_side = "left"
    queries_only_dataset = Dataset.from_pandas(
        dataset.to_pandas()[["query", "prior_answer", "weight_context"]].drop_duplicates(), preserve_index=False
    )
    queries_only_dataset = queries_only_dataset.rename_column(
        "prior_answer", "labels"
    )  # need to make the labels column

    encoded_dataset = queries_only_dataset.map(
        lambda examples: tokenizer(examples["query"], padding=True, return_tensors="pt"),
        batched=True,
        batch_size=batch_sz,
    ).select_columns(["input_ids", "attention_mask", "labels", "weight_context"])
    encoded_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels", "weight_context"], device="cuda"
    )  # required for loading correctly into dataloader
    dataloader = torch.utils.data.DataLoader(encoded_dataset, batch_size=batch_sz)
    predictions, labels, is_correct_all = [], [], []
    num_correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            if hook is not None:
                values = torch.tensor(batch["weight_context"] == 1.0)
                hook.set_binary(values)
            batch.pop("weight_context")
            init_seq_len = batch["input_ids"].shape[1]
            outputs = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
            )
            responses_only = outputs[:, init_seq_len:]
            decoded_responses = tokenizer.batch_decode(responses_only)
            decoded_responses = [r.strip() for r in decoded_responses]
            is_correct = [
                is_response_correct_func(response, label) for response, label in zip(decoded_responses, batch["labels"])
            ]

            num_correct += sum(is_correct)
            total += len(batch["labels"])
            predictions += decoded_responses
            is_correct_all += is_correct
            labels += batch["labels"]

            print(f"Average accuracy at batch {i} (query-only): {num_correct/total} ({num_correct}/{total}).")

    queries_only_dataset = queries_only_dataset.map(
        lambda examples: {
            "predictions": predictions,
            "is_correct": is_correct_all,
        },
        batched=True,  # need to set this so that it sets the predictions column to be one element per row from the list
        batch_size=len(
            queries_only_dataset
        ),  # need to set this so that it doesn't have shape mismatch errors in the length of the column.
    )
    queries_only_df = queries_only_dataset.to_pandas()
    query_to_is_correct = dict(zip(queries_only_df["query"], queries_only_df["is_correct"]))
    query_to_prediction = dict(zip(queries_only_df["query"], queries_only_df["predictions"]))

    return query_to_is_correct, query_to_prediction


def evaluate_model(
    model,
    tokenizer,
    dataset: Dataset,
    max_new_tokens: int = 10,
    batch_sz: int = 8,  # "auto",
    is_response_correct_func=response_startswith_label,
    hook=None,
    feature_collection_hook=None,
):
    """
    Given a dataset with columns ["text", "labels"], generate answers and evaluate model accuracy against those labels.
    1. Generate predictions from text
    2. Extract answer, compare to labels, and return accuracy
    """
    # Free gpu memory
    gc.collect()
    torch.cuda.empty_cache()

    if batch_sz == "auto":
        batch_sz = int(2 * int(sum(get_gpu_memory()) / 1000))
        print(f"Setting batch size to {batch_sz} for eval.")
    tokenizer.padding_side = "left"
    encoded_dataset = dataset.map(
        lambda examples: tokenizer(examples["text"], padding=True, return_tensors="pt"),
        batched=True,
        batch_size=batch_sz,
    ).select_columns(["input_ids", "attention_mask", "labels", "weight_context"])
    print(encoded_dataset[0])
    encoded_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels", "weight_context"], device="cuda"
    )  # required for loading correctly into dataloader
    dataloader = torch.utils.data.DataLoader(encoded_dataset, batch_size=batch_sz)
    predictions, labels, is_correct_all = [], [], []
    num_correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            if hook is not None:
                values = torch.tensor(batch["weight_context"] == 1.0)
                hook.set_binary(values)
            if feature_collection_hook is not None:
                feature_collection_hook.attach(model)
            batch.pop("weight_context")
            init_seq_len = batch["input_ids"].shape[1]
            outputs = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
            )
            responses_only = outputs[:, init_seq_len:]
            decoded_responses = tokenizer.batch_decode(responses_only)
            decoded_responses = [r.strip() for r in decoded_responses]
            is_correct = [
                is_response_correct_func(response, label) for response, label in zip(decoded_responses, batch["labels"])
            ]

            num_correct += sum(is_correct)
            total += len(batch["labels"])
            predictions += decoded_responses
            is_correct_all += is_correct
            labels += batch["labels"]

            print(f"Average accuracy at batch {i}: {num_correct/total} ({num_correct}/{total}).")

    dataset = dataset.map(
        lambda examples: {
            "predictions": predictions,
            "is_correct": is_correct_all,
        },
        batched=True,  # need to set this so that it sets the predictions column to be one element per row from the list
        batch_size=len(
            dataset
        ),  # need to set this so that it doesn't have shape mismatch errors in the length of the column.
    )

    return dataset


def evaluate_model_pscores(
    model,
    tokenizer,
    dataset: Dataset,
    format_func,
    batch_sz: int = 8,  # "auto",
):
    """
    Given a dataset with columns ["text", "labels"], generate answers and evaluate model accuracy against those labels.
    1. Generate predictions from text
    2. Extract answer, compare to labels, and return accuracy
    """
    # import pdb; pdb.set_trace()
    from collections import namedtuple

    context_info = namedtuple("context_info", ["context", "context_weight"])
    # Free gpu memory
    gc.collect()
    torch.cuda.empty_cache()

    if batch_sz == "auto":
        batch_sz = int(2 * int(sum(get_gpu_memory()) / 1000))
        print(f"Setting batch size to {batch_sz} for eval.")
    tokenizer.padding_side = "left"
    contexts = [
        context_info(context=dataset["context"][i], context_weight=dataset["weight_context"][i])
        for i in range(len(dataset))
    ]

    def unpack_sus_and_pscore(example, i):
        sus_score, p_scores = compute_sus_and_persuasion_scores(
            query=example["query"],
            entity=None,
            contexts=contexts,
            format_func=format_func,
            model=model,
            tokenizer=tokenizer,
            answer_map=None,
            bs=batch_sz,
            answer_entity=None,
        )

        return {
            "sus_score": sus_score,
            "p_score": p_scores[
                i
            ],  # the contexts passed in to compute_sus_and_persuasion_scores are in the same order as the rows in the dataset.
        }

    dataset = dataset.map(unpack_sus_and_pscore, with_indices=True)

    return dataset


#########################
# EXPERIMENT MANAGEMENT #
#########################
def get_raw_data_dir(dataset_name: str, subsplit: str):
    return os.path.join(
        "data",
        dataset_name,
        "splits",
        subsplit,
    )


def construct_paths_and_dataset_kwargs(
    DATASET_NAME: str,
    SUBSPLIT: str,
    SEED: int,
    TRAIN_SIZE: int,
    MODEL_ID: str,
    PEFT: bool,
    LORA_MODULES: List[str],
    LOAD_IN_4BIT: bool,
    LOAD_IN_8BIT: bool,
    BATCH_SZ: int,
    GRAD_ACCUM: int,
    NO_TRAIN: bool,
    CONTEXT_WEIGHT_AT_END: bool,
    CONTEXT_WEIGHT_FORMAT: str,
    ANSWER_FORMAT_PROMPT_POSITION: str,
    ADD_ANSWER_FORMAT_PROMPT: bool,
    verbose: bool = False,
):
    DATASET_KWARGS_IDENTIFIABLE = dict(
        seed=SEED,
        train_size=TRAIN_SIZE,
    )
    MODEL_KWARGS_IDENTIFIABLE = dict(
        PEFT=PEFT,
        LORA_MODULES=LORA_MODULES,
        LOAD_IN_4BIT=LOAD_IN_4BIT,
        LOAD_IN_8BIT=LOAD_IN_8BIT,
        BATCH_SZ=BATCH_SZ,
        GRAD_ACCUM=GRAD_ACCUM,
        NO_TRAIN=NO_TRAIN,
    )

    # Paths
    # Construct dataset and data ids
    data_id = f"{DATASET_NAME}_{SUBSPLIT}"
    data_id += f"-ts{TRAIN_SIZE}" if TRAIN_SIZE is not None else ""

    data_dir = os.path.join(
        "data",
        DATASET_NAME,
        data_id,
        f"{SEED}",
    )
    input_dir = os.path.join(data_dir, "inputs")

    raw_data_dir = get_raw_data_dir(
        dataset_name=DATASET_NAME,
        subsplit=SUBSPLIT,
    )
    train_path = os.path.join(raw_data_dir, "train.csv")
    val_path = os.path.join(raw_data_dir, "val.csv")
    test_path = os.path.join(raw_data_dir, "test.csv")

    DATASET_KWARGS_IDENTIFIABLE = {
        **DATASET_KWARGS_IDENTIFIABLE,
        **dict(
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
        ),
    }

    # Check if model id is a path
    if os.path.exists(MODEL_ID):
        # parse only the model id
        MODEL_ID = os.path.basename(MODEL_ID)

    # Construct model id
    model_id = MODEL_ID
    model_id += "-4bit" if MODEL_KWARGS_IDENTIFIABLE["LOAD_IN_4BIT"] else ""
    model_id += "-8bit" if MODEL_KWARGS_IDENTIFIABLE["LOAD_IN_8BIT"] else ""
    if not NO_TRAIN:
        model_id += f"-peft{'_'.join(LORA_MODULES)}" if MODEL_KWARGS_IDENTIFIABLE["PEFT"] else ""
        model_id += f"-bs{MODEL_KWARGS_IDENTIFIABLE['BATCH_SZ']}"
        model_id += (
            f"-ga{MODEL_KWARGS_IDENTIFIABLE['GRAD_ACCUM']}" if MODEL_KWARGS_IDENTIFIABLE["GRAD_ACCUM"] != 1 else ""
        )
        model_id += "-cwe" if CONTEXT_WEIGHT_AT_END else ""
        model_id += f"-cwf_{CONTEXT_WEIGHT_FORMAT}"
        if ADD_ANSWER_FORMAT_PROMPT:
            model_id += f"-afpp_{ANSWER_FORMAT_PROMPT_POSITION}"
    else:
        model_id += "-NT"

    model_parent_dir = os.path.join(data_dir, "models", model_id)
    model_dir = os.path.join(model_parent_dir, "model")

    # Results path
    results_dir = os.path.join(model_parent_dir, "results")
    val_results_path = os.path.join(results_dir, "val.csv")

    if verbose:
        print(f"Data dir: {data_dir}")
        print(f"Model dir: {model_dir}")
        print(f"Results dir: {results_dir}")

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(model_parent_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    return (
        data_dir,
        input_dir,
        model_dir,
        results_dir,
        val_results_path,
        data_id,
        model_id,
        DATASET_KWARGS_IDENTIFIABLE,
        MODEL_KWARGS_IDENTIFIABLE,
    )


def construct_artifact_name(data_id, SEED, model_id, prefix=""):
    artifact_name = f"{data_id}-{SEED}-{model_id}".replace("/", ".")
    artifact_name = prefix + hashlib.sha256(artifact_name.encode()).hexdigest()[:8]
    return artifact_name


def get_gpu_memory() -> List[int]:
    # From https://stackoverflow.com/a/59571639
    # Returns list of MB of free GPU memory per gpu
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


###################
# DATA PROCESSING #
###################
def format_prompts(
    examples: Union[Dataset, dict],
    eos_token: str,
    prompt_template_dict: str,
    demonstrations_context_weight_format: str,
    query_context_weight_format: str,
    context_weight_at_end: bool = False,
    demonstrations_df: pd.DataFrame = pd.DataFrame(),
    do_eval: bool = False,
    answer_format: str = "word",
    add_answer_format_prompt: bool = True,
    answer_format_prompt_position: str = "start",
) -> List[str]:
    """
    Construct a prompt for each example in examples using the prompt_template.

    Args:
        examples - a dataset containing columns ["context", "query", "weight_context", "answer"],
        eos_token - the eos token required to signify the end of the prompt.
        prompt_template - the prompt template for which to fill out with examples
        do_eval - whether to construct the prompt for evaluation mode (True) or training mode (False). For eval mode, the answer is not included in the prompt.
        context_weight_at_end - whether to include the context weight at the end of the context.

    Returns:
        a list of prompts that combines the instruction, formatted input, and expected answer for each example.
    """
    return [
        construct_query_with_demonstrations(
            prompt_template_dict=prompt_template_dict,
            eos_token=eos_token,
            demonstrations_df=demonstrations_df,
            val_context=context,
            val_query=query,
            val_answer=answer,
            context_weight=context_weight,
            demonstrations_context_weight_format=demonstrations_context_weight_format,
            query_context_weight_format=query_context_weight_format,
            context_weight_at_end=context_weight_at_end,
            do_eval=do_eval,
            answer_format=answer_format,
            add_answer_format_prompt=add_answer_format_prompt,
            answer_format_prompt_position=answer_format_prompt_position,
        )
        for (context, query, answer, context_weight) in zip(
            examples["context"], examples["query"], examples["answer"], examples["weight_context"]
        )
    ]


QUERY_TEMPLATE_NO_INSTRUCTION = """Context: {context}
Query: {query}"""

QUERY_TEMPLATE_FLOAT = """Context: {context}
Context weight: {weight:.2f}
Query: {query}"""

QUERY_TEMPLATE_FLOAT_CTX_W_END = """Context: {context}
Query: {query}
Context weight: {weight:.2f}"""

QUERY_TEMPLATE_STR = """Context: {context}
Instruction: {weight}
Query: {query}"""

QUERY_TEMPLATE_STR_CTX_W_END = """Context: {context}
Query: {query}
Instruction: {weight}"""

BASE_TEMPLATE_DICT, BASE_RESPONSE_TEMPLATE = (
    {
        "SYSTEM": "System Instruction: {}\n",
        "ROUND": "User: {}\n\nAssistant: {}",
        "END_OF_ROUND": "\n\n",
    },
    "\n\nAssistant:",
)

# LLAMA3 INSTRUCT
LLAMA3_PROMPT_TEMPLATE_DICT, LLAMA3_RESPONSE_TEMPLATE = (
    {
        "SYSTEM": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>",
        "ROUND": "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{}",
        "END_OF_ROUND": "<|eot_id|>",
    },
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
)  # https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/

# MISTRAL INSTRUCT
MISTRAL_INSTRUCT_PROMPT_TEMPLATE_DICT, MISTRAL_INSTRUCT_RESPONSE_TEMPLATE = (
    {
        "SYSTEM": """<s>[INST] {} \n""",
        "ROUND": """{}[/INST]{}""",
        "END_OF_ROUND": """</s>[INST]""",
    },
    "[/INST]",
)  # https://www.promptingguide.ai/models/mistral-7b#chat-template-for-mistral-7b-instruct

# LLAMA2 CHAT
LLAMA2_PROMPT_TEMPLATE_DICT, LLAMA2_RESPONSE_TEMPLATE = (
    {
        "SYSTEM": """<s>[INST] <<SYS>> {} <</SYS>> \n""",
        "ROUND": """{}[/INST]{}""",
        "END_OF_ROUND": """[INST]""",
    },
    "[/INST]",
)  # https://developer.ibm.com/tutorials/awb-prompt-engineering-llama-2/

# GEMMA
GEMMA_PROMPT_TEMPLATE_DICT, GEMMA_RESPONSE_TEMPLATE = (
    {
        "SYSTEM": """<start_of_turn>user\n{} """,
        "ROUND": """{}<end_of_turn>\n<start_of_turn>model\n{}""",
        "END_OF_ROUND": """<end_of_turn>""",
    },
    "<start_of_turn>model",
)  # https://www.promptingguide.ai/models/gemma#how-to-prompt-gemma-7b

MODEL_ID_TO_TEMPLATES_DICT = {
    "unsloth/llama-3-8b-bnb-4bit": (LLAMA3_PROMPT_TEMPLATE_DICT, LLAMA3_RESPONSE_TEMPLATE),
    "unsloth/llama-3-8b-Instruct-bnb-4bit": (LLAMA3_PROMPT_TEMPLATE_DICT, LLAMA3_RESPONSE_TEMPLATE),
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit": (LLAMA3_PROMPT_TEMPLATE_DICT, LLAMA3_RESPONSE_TEMPLATE),
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit": (LLAMA3_PROMPT_TEMPLATE_DICT, LLAMA3_RESPONSE_TEMPLATE),
    "unsloth/llama-3-8b-bnb-4bit": (LLAMA3_PROMPT_TEMPLATE_DICT, LLAMA3_RESPONSE_TEMPLATE),
    "Meta-Llama-3.1-8B-Instruct": (LLAMA3_PROMPT_TEMPLATE_DICT, LLAMA3_RESPONSE_TEMPLATE),
    "Meta-Llama-3.1-8B": (LLAMA3_PROMPT_TEMPLATE_DICT, LLAMA3_RESPONSE_TEMPLATE),
    "Meta-Llama-3-8B-Instruct": (LLAMA3_PROMPT_TEMPLATE_DICT, LLAMA3_RESPONSE_TEMPLATE),
    "Meta-Llama-3-8B": (BASE_TEMPLATE_DICT, BASE_RESPONSE_TEMPLATE),
    "unsloth/llama-3-8b-bnb-4bit": (LLAMA3_PROMPT_TEMPLATE_DICT, LLAMA3_RESPONSE_TEMPLATE),
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit": (LLAMA3_PROMPT_TEMPLATE_DICT, LLAMA3_RESPONSE_TEMPLATE),
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit": (LLAMA3_PROMPT_TEMPLATE_DICT, LLAMA3_RESPONSE_TEMPLATE),
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit": (
        MISTRAL_INSTRUCT_PROMPT_TEMPLATE_DICT,
        MISTRAL_INSTRUCT_RESPONSE_TEMPLATE,
    ),
    "Mistral-7B-Instruct-v0.3": (
        MISTRAL_INSTRUCT_PROMPT_TEMPLATE_DICT,
        MISTRAL_INSTRUCT_RESPONSE_TEMPLATE,
    ),
    "Mistral-7B-v0.3": (
        MISTRAL_INSTRUCT_PROMPT_TEMPLATE_DICT,
        MISTRAL_INSTRUCT_RESPONSE_TEMPLATE,
    ),
    "unsloth/mistral-7b-v0.3-bnb-4bit": (
        MISTRAL_INSTRUCT_PROMPT_TEMPLATE_DICT,
        MISTRAL_INSTRUCT_RESPONSE_TEMPLATE,
    ),
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit": (
        MISTRAL_INSTRUCT_PROMPT_TEMPLATE_DICT,
        MISTRAL_INSTRUCT_RESPONSE_TEMPLATE,
    ),
    "unsloth/llama-2-7b-chat-bnb-4bit": (LLAMA2_PROMPT_TEMPLATE_DICT, LLAMA2_RESPONSE_TEMPLATE),
    "unsloth/llama-2-7b-bnb-4bit": (LLAMA2_PROMPT_TEMPLATE_DICT, LLAMA2_RESPONSE_TEMPLATE),
    "google/gemma-2-9b": (GEMMA_PROMPT_TEMPLATE_DICT, GEMMA_RESPONSE_TEMPLATE),
    "unsloth/gemma-7b-bnb-4bit": (GEMMA_PROMPT_TEMPLATE_DICT, GEMMA_RESPONSE_TEMPLATE),
    "unsloth/gemma-2b-it-bnb-4bit": (GEMMA_PROMPT_TEMPLATE_DICT, GEMMA_RESPONSE_TEMPLATE),
    "unsloth/gemma-7b-it-bnb-4bit": (GEMMA_PROMPT_TEMPLATE_DICT, GEMMA_RESPONSE_TEMPLATE),
    "unsloth/gemma-2b-it-bnb-4bit": (GEMMA_PROMPT_TEMPLATE_DICT, GEMMA_RESPONSE_TEMPLATE),
    "gemma-2-9b": (GEMMA_PROMPT_TEMPLATE_DICT, GEMMA_RESPONSE_TEMPLATE),
    "gemma-2-9b-it": (GEMMA_PROMPT_TEMPLATE_DICT, GEMMA_RESPONSE_TEMPLATE),
    "unsloth/gemma-2-9b-bnb-4bit": (GEMMA_PROMPT_TEMPLATE_DICT, GEMMA_RESPONSE_TEMPLATE),
    "unsloth/gemma-2-9b-it-bnb-4bit": (GEMMA_PROMPT_TEMPLATE_DICT, GEMMA_RESPONSE_TEMPLATE),
    "openai-community/gpt2": (GEMMA_PROMPT_TEMPLATE_DICT, GEMMA_RESPONSE_TEMPLATE),
}

ANSWER_FORMAT_PROMPT = {
    "word": "Output format: Answer with a single word.",
    "number": "Output format: Answer with a single number.",
}

CTX_WEIGHT_FORMAT_TO_FUNC_AND_QUERY_TEMPLATE = {
    "float": {
        "format_func": lambda ctx_w: ctx_w,
        "query_template": {
            False: QUERY_TEMPLATE_FLOAT,
            True: QUERY_TEMPLATE_FLOAT_CTX_W_END,
        },
    },
    "instruction": {
        "format_func": lambda ctx_w: {
            0: "Ignore the context in answering the query.",
            1: "Only consider the context in answering the query.",
        }[ctx_w],
        "query_template": {
            False: QUERY_TEMPLATE_STR,
            True: QUERY_TEMPLATE_STR_CTX_W_END,
        },
    },
    "none": {
        "format_func": lambda ctx_w: ctx_w,
        "query_template": {
            False: QUERY_TEMPLATE_NO_INSTRUCTION,
            True: QUERY_TEMPLATE_NO_INSTRUCTION,
        },
    },
}  # Given a format type, return (a) a function which will  map a given context weight (as a float) to its string representation AND (b) the query template for that format type.


def create_pscore_format_func(
    prompt_template_dict: Dict[str, str],
    eos_token: str,
    demonstrations_df: pd.DataFrame,  # can be empty
    demonstrations_context_weight_format: str = "float",
    query_context_weight_format: str = "float",
    context_weight_at_end: bool = False,
    answer_format: str = "word",
    add_answer_format_prompt: bool = True,
):
    return lambda query, entity, context: construct_query_with_demonstrations(
        val_query=query,
        val_context=context.context,
        context_weight=context.context_weight,
        val_answer=None,
        prompt_template_dict=prompt_template_dict,
        eos_token=eos_token,
        demonstrations_df=demonstrations_df,
        demonstrations_context_weight_format=demonstrations_context_weight_format,
        query_context_weight_format=query_context_weight_format,
        context_weight_at_end=context_weight_at_end,
        do_eval=True,
        answer_format=answer_format,
        add_answer_format_prompt=add_answer_format_prompt,
    )


def construct_query_with_demonstrations(
    prompt_template_dict: Dict[str, str],
    eos_token: str,
    demonstrations_df: pd.DataFrame,  # can be empty
    val_context: str,
    val_query: str,
    val_answer: str,
    context_weight: int = 1.0,
    demonstrations_context_weight_format: str = "float",
    query_context_weight_format: str = "float",
    context_weight_at_end: bool = False,
    do_eval: bool = False,
    answer_format: str = "word",
    add_answer_format_prompt: bool = False,
    answer_format_prompt_position: str = "start",
) -> str:
    if demonstrations_context_weight_format is None:
        demonstrations_context_weight_format = query_context_weight_format
    return (
        construct_system_prompt(prompt_template_dict)
        + construct_demonstrations(
            prompt_template_dict=prompt_template_dict,
            eos_token=eos_token,
            demonstrations_df=demonstrations_df,  # can be empty
            context_weight_format=demonstrations_context_weight_format,
            context_weight_at_end=context_weight_at_end,
            answer_format=answer_format,
            add_answer_format_prompt=add_answer_format_prompt,
            answer_format_prompt_position=answer_format_prompt_position,
        )
        + construct_query(
            prompt_template_dict=prompt_template_dict,
            eos_token=eos_token,
            val_context=val_context,
            val_query=val_query,
            val_answer=val_answer,
            context_weight=context_weight,
            context_weight_format=query_context_weight_format,
            context_weight_at_end=context_weight_at_end,
            do_eval=do_eval,
            answer_format=answer_format,
            add_answer_format_prompt=add_answer_format_prompt,
            answer_format_prompt_position=answer_format_prompt_position,
        )
    )


def construct_system_prompt(prompt_template_dict):
    return prompt_template_dict["SYSTEM"].format(
        "Answer the following query considering the provided context. Answer with only one word."
    )


def construct_demonstrations(
    prompt_template_dict: Dict[str, str],
    eos_token: str,
    demonstrations_df: pd.DataFrame,  # can be empty
    context_weight_format: Optional[str] = "float",
    context_weight_at_end: bool = False,
    answer_format: str = "word",
    add_answer_format_prompt: bool = True,
    answer_format_prompt_position: str = "start",
):
    if context_weight_format is None:
        if len(demonstrations_df) > 0:
            raise ValueError(
                "context weight format for demonstrations is None but demonstrations_df is not empty. Either remove the demonstrations or specify how to format them."
            )
        else:
            return ""

    format_ctx_weight_func = CTX_WEIGHT_FORMAT_TO_FUNC_AND_QUERY_TEMPLATE[context_weight_format]["format_func"]
    query_template = CTX_WEIGHT_FORMAT_TO_FUNC_AND_QUERY_TEMPLATE[context_weight_format]["query_template"][
        context_weight_at_end
    ]

    # Construct the demontrations into the string (if they exist)
    rounds = []
    for i, row in demonstrations_df.iterrows():
        query = query_template.format(
            context=row["context"], weight=format_ctx_weight_func(row["weight_context"]), query=row["query"]
        )
        if add_answer_format_prompt:
            if answer_format_prompt_position == "start":
                query = f"{ANSWER_FORMAT_PROMPT[answer_format]}\n{query}"
            else:
                query = f"{query}\n{ANSWER_FORMAT_PROMPT[answer_format]}"
        round = prompt_template_dict["ROUND"].format(query, row["answer"])
        round += prompt_template_dict["END_OF_ROUND"]
        rounds.append(round)

    return "".join(rounds)


def construct_query(
    prompt_template_dict: Dict[str, str],
    eos_token: str,
    val_context: str,
    val_query: str,
    val_answer: str,
    answer_format: str = "word",
    context_weight: int = 1.0,
    context_weight_format: str = "float",
    context_weight_at_end: bool = False,
    do_eval: bool = False,
    add_answer_format_prompt: bool = True,
    answer_format_prompt_position: str = "start",
):
    format_ctx_weight_func = CTX_WEIGHT_FORMAT_TO_FUNC_AND_QUERY_TEMPLATE[context_weight_format]["format_func"]
    query_template = CTX_WEIGHT_FORMAT_TO_FUNC_AND_QUERY_TEMPLATE[context_weight_format]["query_template"][
        context_weight_at_end
    ]
    query = query_template.format(context=val_context, weight=format_ctx_weight_func(context_weight), query=val_query)
    if add_answer_format_prompt:
        if answer_format_prompt_position == "start":
            query = f"{ANSWER_FORMAT_PROMPT[answer_format]}\n{query}"
        else:
            query = f"{query}\n{ANSWER_FORMAT_PROMPT[answer_format]}"
    return prompt_template_dict["ROUND"].format(
        query,
        "" if do_eval else val_answer + prompt_template_dict["END_OF_ROUND"] + eos_token,
        # Must add EOS_TOKEN during training, otherwise your generation will go on forever!
    )


def sample_few_shot_examples(train_df: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    """
    Assume that train_df contains 0/1 context weight examples adjacent to each other.
    k - total number of few shot examples (k / 2 pairs)
    """
    shot_indices = train_df[::2].sample(k // 2, random_state=seed).index
    shot_indices = [(i, i + 1) for i in shot_indices]
    shot_indices = np.array(shot_indices).flatten()
    shot_sample = train_df.loc[shot_indices]
    return shot_sample


ALPACA_PROMPT, ALPACA_RESPONSE_TEMPLATE = (
    """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}""",
    "Response:",
)

GEMMA_PROMPT, GEMMA_RESPONSE_TEMPLATE = (
    """<start_of_turn>user
{}

{}<end_of_turn>
<start_of_turn>model
{}""",
    "<start_of_turn>model",
)  # https://www.promptingguide.ai/models/gemma#how-to-prompt-gemma-7b

GPT2_PROMPT, GPT2_RESPONSE_TEMPLATE = (
    """{}
Q: {}
A: {}""",
    "A:",
)

PHI_PROMPT, PHI_RESPONSE_TEMPLATE = (
    """Instruct: {}
{}
Output: {}""",
    "Output:",
)

MISTRAL_INSTRUCT_PROMPT, MISTRAL_INSTRUCT_RESPONSE_TEMPLATE = (
    "<s>[INST] {}\n{} [/INST] {}",
    "[/INST] ",
)  # https://www.promptingguide.ai/models/mistral-7b#chat-template-for-mistral-7b-instruct

LLAMA2_PROMPT, LLAMA2_RESPONSE_TEMPLATE = (
    """<s>[INST] <<SYS>>
{}
<</SYS>>

{} [/INST]{}
""",
    "[/INST]",
)  # https://developer.ibm.com/tutorials/awb-prompt-engineering-llama-2/

LLAMA3_PROMPT, LLAMA3_RESPONSE_TEMPLATE = (
    """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{}<|eot_id|><|start_header_id|>user<|end_header_id|>

{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{}
""",
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
)  # https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/

PROMPTS_DICT = {
    "unsloth/mistral-7b-v0.2-bnb-4bit": (ALPACA_PROMPT, ALPACA_RESPONSE_TEMPLATE),
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit": (MISTRAL_INSTRUCT_PROMPT, MISTRAL_INSTRUCT_RESPONSE_TEMPLATE),
    "unsloth/llama-2-7b-bnb-4bit": (LLAMA2_PROMPT, LLAMA2_RESPONSE_TEMPLATE),
    "unsloth/llama-2-7b-chat-bnb-4bit": (LLAMA2_PROMPT, LLAMA2_RESPONSE_TEMPLATE),
    "unsloth/llama-3-8b-bnb-4bit": (LLAMA3_PROMPT, LLAMA3_RESPONSE_TEMPLATE),
    "unsloth/llama-3-8b-Instruct-bnb-4bit": (LLAMA3_PROMPT, LLAMA3_RESPONSE_TEMPLATE),
    "unsloth/gemma-2b-bnb-4bit": (GEMMA_PROMPT, GEMMA_RESPONSE_TEMPLATE),
    "unsloth/gemma-7b-bnb-4bit": (GEMMA_PROMPT, GEMMA_RESPONSE_TEMPLATE),
    "unsloth/gemma-2b-it-bnb-4bit": (GEMMA_PROMPT, GEMMA_RESPONSE_TEMPLATE),
    "unsloth/gemma-7b-it-bnb-4bit": (GEMMA_PROMPT, GEMMA_RESPONSE_TEMPLATE),
    "openai-community/gpt2": (GPT2_PROMPT, GPT2_RESPONSE_TEMPLATE),
    "microsoft/phi-1_5": (PHI_PROMPT, PHI_RESPONSE_TEMPLATE),
}


from typing import NamedTuple


def construct_test_results_dir(
    base_results_dir: str,
    eval_name: str,
    subsplit: str,
    k_demonstrations: int,
    in_domain_demonstrations: bool,
    context_weight_format: str,
    answer_format_prompt_position: str,
    add_answer_format_prompt: bool,
    do_steering: bool,
    steering_prior_value: float,
    steering_context_value: float,
    steering_layer: str,
):
    eval_id = eval_name
    eval_id += f"-sp_{subsplit}"
    eval_id += f"-k{k_demonstrations}" + ("_ID" if in_domain_demonstrations else "_OOD")
    eval_id += f"-cwf_{context_weight_format}"
    if add_answer_format_prompt:
        eval_id += f"-afpp_{answer_format_prompt_position}"
    if do_steering:
        eval_id += f"-steer_l{steering_layer}_p{steering_prior_value}_c{steering_context_value}"
    return os.path.join(base_results_dir, eval_id)


class EvalConfig(NamedTuple):
    """Config for evaluating a model's ability to follow context vs prior according to a weight flag."""

    dataset_name: str
    subsplit: str
    k_demonstrations: int
    context_weight_format: str
    do_steering: bool = False

def validate_extracted_tokens_match_names(extracted_toks: torch.LongTensor, ex_ids_expanded: List[str], ex_ids_to_ch_name: Dict[int, str], tokenizer):
    """Validate the extracted tokens actually are those of the protagonist names."""
    if len(extracted_toks) != len(ex_ids_expanded):
        raise ValueError(f"Expected ex_ids_expanded ({ex_ids_expanded.shape}) to have same length as extracted_toks ({extracted_toks.shape}).")
    for i, tok in enumerate(extracted_toks.tolist()):
        pro_name = ex_ids_to_ch_name[ex_ids_expanded[i, 0].item()]
        valid_name_list = {pro_name, " " + pro_name}
        if tokenizer.decode(tok) not in valid_name_list:
            raise ValueError(f"Extracted tok {tokenizer.decode(tok)} is not in valid_name_list {valid_name_list}. Check that extraction is done correctly.")

def validate_extracted_tokens_match(extracted_toks: torch.LongTensor, tokenizer, valid_tok_list=["."]):
    """Validate the extracted tokens actually are those expected of the protagonist names."""
    for tok in extracted_toks.tolist():
        if tokenizer.decode(tok) not in valid_tok_list:
            raise ValueError(f"Extracted tok {tokenizer.decode(tok)} is not in valid_name_list {valid_tok_list}. Check that extraction is done correctly.")

def get_residuals(model, tokenizer, prompts, resid_type="all", batch_size=32) -> Tuple[torch.FloatTensor, torch.LongTensor]:
    """
    Gather residuals using transformers built-in hidden states output.
    
    Args:
        model: The LLaMA model
        tokenizer: The tokenizer
        prompts: List of text prompts
        batch_size: Batch size for processing
    
    Returns:
        List of hidden states tensors for each prompt, and the number of pads for each example
    """
    all_hidden_states = []
    all_num_pads = []
    # Tokenize
    inputs = tokenizer(
        prompts,
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(model.device)
    print("Device:", model.device)
    
    # Process prompts in batches
    for i in tqdm(range(0, len(prompts), batch_size)):
        print(f"Processing batch {i // batch_size} of {len(prompts) // batch_size}")
        tokens = inputs["input_ids"][i:i + batch_size] # shape: (bs, T)
        attn_mask = inputs["attention_mask"][i:i + batch_size] # shape: (bs, T)
        num_pads = (~attn_mask.bool()).sum(dim=1) # shape: (bs,)
        
        # Forward pass with hidden states output
        with torch.no_grad():
            outputs = model(tokens, attn_mask, output_hidden_states=True)
            
        # outputs.hidden_states is a tuple with hidden states from each layer
        # The last element is the final hidden state
        if resid_type == "last_layer":
            hidden_states = outputs.hidden_states[-1]  # Shape: (batch_size, seq_len, hidden_size)
        elif resid_type == "last_token":
            hidden_states = torch.stack([hs[:, -1, :] for hs in outputs.hidden_states], dim=1) # shape: (bs, # layers, hidden size)
        elif resid_type == "all":
            hidden_states = []
            for state in outputs.hidden_states:
                hidden_states.append(state.cpu())
            hidden_states = torch.stack(hidden_states, dim=2)
        else:
            raise ValueError("expected either `last_layer` or `last_token`.")
        all_hidden_states.append(hidden_states)
        all_num_pads.append(num_pads)

    return torch.cat(all_hidden_states, dim=0).float(), torch.cat(all_num_pads, dim=0).int(), inputs["input_ids"].cpu()

def get_story_detail_idx(prompt: str, phrase: str, tokenizer, phrase_token_offset=-1):
    phrase_tokens = tokenizer.encode(phrase, add_special_tokens=False)  # Add space prefix since it's mid-sentence
    print("Phrase tokens:", [tokenizer.decode(t) for t in phrase_tokens])
    # Get tokens for the full prompt and find where this sequence appears
    prompt_tokens = tokenizer(prompt)["input_ids"]
    prompt_decoded = [tokenizer.decode(t) for t in prompt_tokens]

    # Find the position of these tokens in the full sequence
    start_idx = None
    for i in range(len(prompt_tokens) - len(phrase_tokens) + 1):
        if prompt_tokens[i:i+len(phrase_tokens)] == phrase_tokens:
            start_idx = i
            break

    if start_idx is not None:
        print(f"\nFound phrase starting at position {start_idx}")
        print("Context:")
        for i in range(max(0, start_idx-2), min(len(prompt_decoded), start_idx + len(phrase_tokens) + 12)):
            print(f"{i}: {prompt_decoded[i]}")

    start_idx = start_idx + phrase_token_offset
    print("Start idx:", start_idx)
    return start_idx

def extract_character_name(prompt: str, phrase: str = " puts a ball under") -> str:
    """Extract the name of the character who performs an action.
    
    Args:
        prompt: String containing text like "John puts a ball under" or "Kate puts a ball under"
        phrase: The action phrase that follows the character name (default: "puts a ball under")
        
    Returns:
        The extracted name (e.g. "John" or "Kate") or None if no match found
    """
    import re
    pattern = rf'(\w+){re.escape(phrase)}'
    match = re.search(pattern, prompt)
    if match:
        return match.group(1)
    return None

def get_story_start_idx(prompt: str, tokenizer):
    phrase = " puts a ball under"
    name = extract_character_name(prompt, phrase)
    tokens_in_name = tokenizer.encode(name, add_special_tokens=False)
    print("Tokens in name:", [tokenizer.decode(t) for t in tokens_in_name])

    return get_story_detail_idx(prompt=prompt, phrase=phrase, tokenizer=tokenizer, phrase_token_offset=-len(tokens_in_name))

def extract_token_labels(residuals: torch.FloatTensor, labels: List[List[Tuple[int, int, int]]], ex_ids: torch.LongTensor, num_pads: torch.LongTensor, corresponding_tokens: torch.LongTensor):
    """
    Args:
        residuals: shape (num_examples, num_tokens, num_layers, hidden_dim)
        labels: list of list of tuples (token_pos, label, mention_idx). 
               Each outer list corresponds to a single example. 
               Each inner list corresponds to the (token positions, label, mention_idx) for each name mention in that example.
        ex_ids: tensor of shape (num_examples,) containing example IDs
        corresponding_tokens: tensor of shape (num_examples, num_tokens)
    Returns:
        masked_residuals: shape (num_resids, num_layers, hidden_dim)
        masked_labels: shape (num_resids,) 
        masked_ex_info: shape (num_resids, 2) containing (ex_id, mention_idx) for each residual
        where num_resids is the number of valid residuals across all examples (name occurrences)
    """
    num_examples = len(residuals)
    num_tokens = residuals.size(1)
    device = residuals.device
    
    # Create tensors to hold labels and mention indices for each token position
    token_labels = torch.zeros((num_examples, num_tokens), dtype=torch.long, device=device)
    mention_indices = torch.zeros((num_examples, num_tokens), dtype=torch.long, device=device)
    
    # Create a mask for valid positions (positions where names occur)
    valid_positions = torch.zeros((num_examples, num_tokens), dtype=torch.bool, device=device)
    
    # Fill in the labels and mention indices based on the token positions
    for i, example_labels in enumerate(labels):
        for pos, label, mention_idx in example_labels:
            pad_shifted_pos = pos + num_pads[i]
            token_labels[i, pad_shifted_pos] = label
            mention_indices[i, pad_shifted_pos] = mention_idx
            valid_positions[i, pad_shifted_pos] = True
    
    # Create a tensor of example IDs expanded to match token dimensions
    ex_ids_expanded = ex_ids.unsqueeze(1).expand(-1, num_tokens)
    
    # Apply mask to residuals and labels
    masked_residuals = residuals[valid_positions]  # Will flatten automatically
    masked_labels = token_labels[valid_positions]  # Will flatten automatically
    masked_tokens = corresponding_tokens[valid_positions]

    # Create masked example info combining ex_ids and mention indices
    masked_ex_ids = ex_ids_expanded[valid_positions]
    masked_mention_indices = mention_indices[valid_positions]
    masked_ex_info = torch.stack([masked_ex_ids, masked_mention_indices], dim=1)
    
    return masked_residuals, masked_labels, masked_ex_info, masked_tokens


def get_residuals_and_labels(
    model, 
    tokenizer, 
    prompts, 
    awareness_labels, 
    pb_labels,
    token_types,
    ex_ids, 
    batch_size=32, 
    resid_type="all"
):
    """
    Combined function to get residuals and extract token labels in a memory-efficient way.
    
    Args:
        model: The language modeltoken_types
        tokenizer: The tokenizer
        prompts: List of text prompts
        awareness_labels: List of list of tuples (token_pos, label, mention_idx) for awareness
        pb_labels: List of list of tuples (token_pos, label, mention_idx) for protagonist belief
        ex_ids: tensor of shape (num_examples,) containing example IDs
        batch_size: Batch size for processing
        resid_type: Type of residuals to extract ("all", "last_layer", or "last_token")
    
    Returns:
        masked_residuals: shape (total_num_resids, num_layers, hidden_dim)
        masked_awareness_labels: shape (total_num_resids,)
        masked_pb_labels: shape (total_num_resids,)
        masked_ex_info: shape (total_num_resids, 2) containing (ex_id, mention_idx)
        masked_corresponding_tokens: shape (total_num_resids,)
    """
    all_masked_residuals = []
    all_masked_awareness_labels = []
    all_masked_pb_labels = []
    all_masked_token_types = []
    all_masked_ex_info = []
    all_masked_corresponding_tokens = []

    device = model.device

    # Tokenize batch
    inputs = tokenizer(
        prompts,
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)
    
    # Process prompts in batches
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_end = min(i + batch_size, len(prompts))
        print(f"Processing batch {i//batch_size + 1} of {(len(prompts) + batch_size - 1)//batch_size}")

        batch_tokens = inputs["input_ids"][i:batch_end]
        batch_attn_mask = inputs["attention_mask"][i:batch_end]
        batch_awareness_labels = awareness_labels[i:batch_end]
        batch_pb_labels = pb_labels[i:batch_end]
        batch_token_types = token_types[i:batch_end]
        batch_ex_ids = ex_ids[i:batch_end]

        # from nnsight import LanguageModel
        # if isinstance(model, LanguageModel): 
        #     residuals = []
        #     with model.session(remote=True) as session:
        #         with model.trace("The Eiffel Tower is in the city of", remote=True) as runner:
        #             for layer in model.model.layers:
        #                 residuals.append(layer.output[0].save())


        # else:
        # Get residuals for batch
        with torch.no_grad():
            outputs = model(
                batch_tokens, 
                batch_attn_mask, 
                output_hidden_states=True
            )
            
        # Process hidden states based on resid_type
        if resid_type == "last_layer":
            hidden_states = outputs.hidden_states[-1]
        elif resid_type == "last_token":
            hidden_states = torch.stack([hs[:, -1, :] for hs in outputs.hidden_states], dim=1)
        elif resid_type == "all":
            hidden_states = []
            for state in outputs.hidden_states:
                hidden_states.append(state.cpu())
            hidden_states = torch.stack(hidden_states, dim=2)
        else:
            raise ValueError("expected either `last_layer`, `last_token`, or `all`")

        # Extract token labels for batch
        num_batch_examples = batch_end - i
        num_tokens = hidden_states.size(1)
        
        # Create tensors for batch
        token_awareness_labels = torch.zeros((num_batch_examples, num_tokens), dtype=torch.long, device=hidden_states.device)
        token_pb_labels = torch.zeros((num_batch_examples, num_tokens), dtype=torch.long, device=hidden_states.device)
        # token_token_types = -torch.ones((num_batch_examples, num_tokens), dtype=torch.long, device=hidden_states.device)
        token_token_types = np.empty((num_batch_examples, num_tokens), dtype=object)
        mention_indices = torch.zeros((num_batch_examples, num_tokens), dtype=torch.long, device=hidden_states.device)
        valid_positions = torch.zeros((num_batch_examples, num_tokens), dtype=torch.bool, device=hidden_states.device)
        
        # Fill in labels and mention indices
        for j, (example_awareness_labels, example_pb_labels, example_token_types) in enumerate(zip(batch_awareness_labels, batch_pb_labels, batch_token_types)):
            # Verify that awareness and pb labels refer to same token positions
            awareness_positions = [x[0] for x in example_awareness_labels]
            pb_positions = [x[0] for x in example_pb_labels]
            if awareness_positions != pb_positions:
                raise ValueError(f"Awareness and PB labels have different token positions: {awareness_positions} vs {pb_positions}")
                
            for (pos, awareness_label, mention_idx), (_, pb_label, _), tok_type in zip(example_awareness_labels, example_pb_labels, example_token_types):
                pad_shifted_pos = pos + (~batch_attn_mask[j].bool()).sum()
                token_awareness_labels[j, pad_shifted_pos] = awareness_label
                token_pb_labels[j, pad_shifted_pos] = pb_label
                mention_indices[j, pad_shifted_pos] = mention_idx
                token_token_types[j, pad_shifted_pos] = tok_type.value
                valid_positions[j, pad_shifted_pos] = True
        
        # Create expanded ex_ids for batch
        batch_ex_ids_expanded = batch_ex_ids.unsqueeze(1).expand(-1, num_tokens)
        
        # Apply mask to batch
        batch_masked_residuals = hidden_states[valid_positions]
        batch_masked_awareness_labels = token_awareness_labels[valid_positions]
        batch_masked_pb_labels = token_pb_labels[valid_positions]
        batch_masked_token_token_types = token_token_types[valid_positions]
        batch_masked_ex_ids = batch_ex_ids_expanded[valid_positions]
        batch_masked_mention_indices = mention_indices[valid_positions]
        batch_masked_ex_info = torch.stack([batch_masked_ex_ids, batch_masked_mention_indices], dim=1)
        batch_masked_corresponding_tokens = batch_tokens[valid_positions]
        
        # Append batch results
        all_masked_residuals.append(batch_masked_residuals.cpu())
        all_masked_awareness_labels.append(batch_masked_awareness_labels.cpu())
        all_masked_pb_labels.append(batch_masked_pb_labels.cpu())
        all_masked_token_types.append(batch_masked_token_token_types)
        all_masked_ex_info.append(batch_masked_ex_info.cpu())
        all_masked_corresponding_tokens.append(batch_masked_corresponding_tokens.cpu())

        # Clear CUDA cache after each batch
        torch.cuda.empty_cache()
    
    # Concatenate all batches
    final_masked_residuals = torch.cat(all_masked_residuals, dim=0).float()
    final_masked_awareness_labels = torch.cat(all_masked_awareness_labels, dim=0)
    final_masked_pb_labels = torch.cat(all_masked_pb_labels, dim=0)
    final_masked_token_types = np.concat(all_masked_token_types, axis=0)
    final_masked_ex_info = torch.cat(all_masked_ex_info, dim=0)
    final_masked_corresponding_tokens = torch.cat(all_masked_corresponding_tokens, dim=0)
    
    return (
        final_masked_residuals,
        final_masked_awareness_labels,
        final_masked_pb_labels,
        final_masked_token_types,
        final_masked_ex_info,
        final_masked_corresponding_tokens
    )

def tok_idx_in_query_sentence(tok_idx, tokens, tokenizer):
    """
    A query sentence is one that ends with a question mark.
    
    Args:
        tok_idx: index of the token to check
        tokens: list of tokens
        tokenizer: tokenizer
    Returns:
        True if the sentence containing tok_idx ends with a question mark, False otherwise
    """
    question_token = tokenizer.encode("?", add_special_tokens=False)[0]
    period_token = tokenizer.encode(".", add_special_tokens=False)[0]
    while tok_idx < len(tokens) and tokens[tok_idx] not in {period_token, question_token}:
        tok_idx += 1

    if tok_idx == len(tokens):
        raise ValueError(f"Token index {tok_idx} is not in a sentence ending with a question mark or period.")

    return tokens[tok_idx] == question_token

def model_response_formatting(ball_location, reasoning, tom=False, cot=True, few_shot=None, answer_start=""):
    assert len(answer_start) == 0 or cot, "Cannot use answer_start with chain-of-thought"
    if cot:
        return reasoning + f"## Result: {answer_start}{ball_location}\n"
    else:
        return answer_start + str(ball_location)

QUERY_TYPE_TO_SP_QUERY_TYPE = {
    "pro_belief": "state",
    "ant_belief": "state",
    "actual": "state",
    "pro_awareness": "awareness",
    "pro_belief_question": "state",
    "pro_belief_believes": "state",
    "pro_belief_predicts": "state",
    "pro_belief_action": "state",
    "pro_belief_action2": "state",
    "pro_belief_action_q": "state",
}
DATASET_NAME_TO_TASK_MAP = {
    "SameSentencesBallToMDataset": "ball",
    "BasicBallToMDataset": "ball",
    "unexpected_contents": "unexpected_contents",
    "unexpected_transfer": "unexpected_transfer",
    "ball": "ball",
}
SYSTEM_PROMPT_MAP = {
    "state": { # SP Query type
        "tom_simple": # SP type
            {
                "unexpected_contents": "You are an expert in theory of mind. Complete the following story. Only answer the next phrase to complete the sentence.",
                "unexpected_transfer":  "You are an expert in theory of mind. Complete the following story. Only answer the next phrase to complete the sentence.",
                "ball":  "You are an expert in theory of mind. Answer the question in the following story. Only answer with the result.",
            },
        "tom": 
            {
                "unexpected_contents": "You are an expert in theory of mind. Your goal is to track the contents of the container in the following story, as well as what each character believes is in the container. Complete the following story. Only answer the next phrase to complete the sentence.",
                "unexpected_transfer":  "You are an expert in theory of mind. Your goal is to track the location of the item of interest, as well as where each character believes the item of the interest is. Complete the following story. Only answer the next phrase to complete the sentence.",
                "ball":  "You are an expert in theory of mind. Your goal is to track where the ball is in the following story, as well as where each character believes the ball is. Answer the question in the following story. Only answer with the result.",
            },
        "reasoning": {
            "unexpected_contents": "You are an expert in logical reasoning. Your goal is to track where the the contents of the container is in the following story.",
            "unexpected_transfer": "You are an expert in logical reasoning. Your goal is to track the location of the item of interest in the following story.",
            "ball": "You are an expert in logical reasoning. Your goal is to track where the ball is in the following story.",
        },
        "reasoning_before": {
            "ball": "Answer the question in the following story. Think step-by-step. Give your final answer in the form '## Result: <number>', such as '## Result: 8'.",
        },
        "reasoning_after": {
            "ball": "Answer the question in the following story. First, give your answer in the form '## Result: <number>', such as '## Result: 8'. Then, explain why this is your answer.",
        },
        "base": {
            "unexpected_contents": "Complete the following story. Only answer the next phrase to complete the sentence.",
            "unexpected_transfer":  "Complete the following story. Only answer the next phrase to complete the sentence.",
            "ball":  "Answer the question in the following story. Only answer with the number.",
        },
        "most_base": {
            "unexpected_contents": "Complete the following story. Only answer the next phrase to complete the sentence.",
            "unexpected_transfer":  "Complete the following story. Only answer the next phrase to complete the sentence.",
            "ball":  "Answer the question in the following story. Give your answer in the form '## Result: <number>', such as '## Result: 8'.",
        },
        "desc_belief": {
            "unexpected_contents": "Complete the following story. Only answer the next phrase to complete the sentence.",
            "unexpected_transfer":  "Complete the following story. Only answer the next phrase to complete the sentence.",
            "ball":  "Describe each character's belief in the following story.",
        },
    },
    "awareness": {
        "base":{
            "ball": "Answer the question in the following story. Only answer with '1' if the answer is Yes, or '0' if the answer is No.",
        },
        "most_base":{
            "ball": "Answer the question in the following story. Answer with '1' if the answer is Yes, or '0' if the answer is No. Give your answer in the form '## Result: <number>', such as '## Result: 1'.",
        },
        "reasoning_before": {
            "ball": "Answer the question in the following story. Think step-by-step. Give your final answer in the form '## Result: <number>', such as '## Result: 8'.",
        },
        "reasoning_after": {
            "ball": "Answer the question in the following story. First, give your answer in the form '## Result: <number>', such as '## Result: 8'. Then, explain why this is your answer.",
        },
    }
}


def is_instruct(tokenizer):
    return "Instruct" in tokenizer.name_or_path or "phi" in tokenizer.name_or_path

def apply_base_template(text, tokenizer):
    import os
    if "Llama-3" in os.path.basename(tokenizer.name_or_path):
        return f"<|begin_of_text|>Answer the question with Yes or No. Q: {text}\nA:"
    elif "DeepSeek-R1-Distill-Llama-70B" in tokenizer.name_or_path:
        return "<|begin_of_text|>" + text
    elif "Qwen3" in tokenizer.name_or_path:
        return f"<|im_start|>system\n/no_think Answer the question with Yes or No.<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\nA:"
    elif "gemma" in tokenizer.name_or_path:
        return f"<start_of_turn>user\nAnswer the question with Yes or No. Q: {text}<end_of_turn>\n<start_of_turn>model\nA:"
    else:
        raise NotImplementedError(f"Base template for {os.path.basename(tokenizer.name_or_path)} not yet implemented.")

def to_chat_template(text, tokenizer, system_prompt_type="base"):
    
    if is_instruct(tokenizer):
        system = "Answer the question with Yes or No."
        rounds = [
            {
                "role": "system",
                "content": system
            }
        ]
        
        rounds.append({
            "role": "user",
            "content": text
        })
        
        text = tokenizer.apply_chat_template(rounds, tokenize=False, add_generation_prompt=True)
        return text
    else:
        text = apply_base_template(text, tokenizer)
        return text
