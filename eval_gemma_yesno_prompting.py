import argparse
import re
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class Example:
    assertion: Optional[str]
    question: str
    expected_answer: str  # "yes" or "no"


CONTEXT_EXAMPLES: list[Example] = [
    Example(
        assertion="In this context, the capital of France is London.",
        question="Is the capital of France London?",
        expected_answer="yes",
    ),
    Example(
        assertion="In this context, the largest planet is Earth.",
        question="Is the largest planet Earth?",
        expected_answer="yes",
    ),
    Example(
        assertion="In this context, water freezes at 10 degrees Celsius.",
        question="Does water freeze at 10 degrees Celsius?",
        expected_answer="yes",
    ),
    Example(
        assertion="In this context, the chemical symbol for gold is Ag.",
        question="Is the chemical symbol for gold Ag?",
        expected_answer="yes",
    ),
    Example(
        assertion="In this context, Tokyo is in South Korea.",
        question="Is Tokyo in South Korea?",
        expected_answer="yes",
    ),
    Example(
        assertion="In this context, the capital of Italy is Milan.",
        question="Is the capital of Italy Rome?",
        expected_answer="no",
    ),
    Example(
        assertion="In this context, the Sun rises in the west.",
        question="Does the Sun rise in the east?",
        expected_answer="no",
    ),
    Example(
        assertion="In this context, humans have three lungs.",
        question="Do humans have two lungs?",
        expected_answer="no",
    ),
    Example(
        assertion="In this context, the Pacific Ocean is the smallest ocean.",
        question="Is the Pacific Ocean the largest ocean?",
        expected_answer="no",
    ),
    Example(
        assertion="In this context, 2 + 2 equals 5.",
        question="Does 2 + 2 equal 4?",
        expected_answer="no",
    ),
]


QUERY_ONLY_EXAMPLES: list[Example] = [
    Example(
        assertion=None,
        question="Is the capital of France Paris?",
        expected_answer="yes",
    ),
    Example(
        assertion=None,
        question="Is 2 + 2 equal to 4?",
        expected_answer="yes",
    ),
    Example(
        assertion=None,
        question="Is water H2O?",
        expected_answer="yes",
    ),
    Example(
        assertion=None,
        question="Is the Sun a star?",
        expected_answer="yes",
    ),
    Example(
        assertion=None,
        question="Does Earth orbit the Sun?",
        expected_answer="yes",
    ),
    Example(
        assertion=None,
        question="Is the capital of Italy Milan?",
        expected_answer="no",
    ),
    Example(
        assertion=None,
        question="Is Tokyo in China?",
        expected_answer="no",
    ),
    Example(
        assertion=None,
        question="Is 10 smaller than 3?",
        expected_answer="no",
    ),
    Example(
        assertion=None,
        question="Is gold's chemical symbol Fe?",
        expected_answer="no",
    ),
    Example(
        assertion=None,
        question="Do humans have three hearts?",
        expected_answer="no",
    ),
]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_yes_no_prefix(text: str) -> Optional[str]:
    m = re.match(r"^\s*(yes|no)\b", text.strip(), flags=re.IGNORECASE)
    return m.group(1).lower() if m else None


def build_messages(style: str, ex: Example, suite_name: str) -> list[dict[str, str]]:
    if suite_name == "context":
        user_payload = f"{ex.assertion}\n\nQuestion: {ex.question}"
    elif suite_name == "query_only":
        user_payload = f"Question: {ex.question}"
    else:
        raise ValueError(f"Unknown suite name: {suite_name}")

    if style == "system_then_user":
        return [
            {"role": "system", "content": "Answer the question with Yes or No only."},
            {"role": "user", "content": user_payload},
        ]

    if style == "user_only_directive":
        user_text = (
            "Answer with Yes or No only.\n"
            f"{user_payload}"
        )
        return [{"role": "user", "content": user_text}]

    if style == "few_shot":
        if suite_name == "context":
            shots = [
                {"role": "user", "content": "In this context, snow is black.\n\nQuestion: Is snow black?"},
                {"role": "assistant", "content": "Yes"},
                {"role": "user", "content": "In this context, cats are plants.\n\nQuestion: Are cats animals?"},
                {"role": "assistant", "content": "No"},
            ]
        else:
            shots = [
                {"role": "user", "content": "Question: Is snow white?"},
                {"role": "assistant", "content": "Yes"},
                {"role": "user", "content": "Question: Is fire cold?"},
                {"role": "assistant", "content": "No"},
            ]
        return [
            {"role": "system", "content": "Answer the question with Yes or No only."},
            *shots,
            {"role": "user", "content": user_payload},
        ]

    raise ValueError(f"Unknown style: {style}")


def generate_answer(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    max_new_tokens: int,
) -> str:
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def print_tokenized_prompt_example(
    tokenizer,
    style: str,
    suite_name: str,
    example: Example,
) -> None:
    messages = build_messages(style, example, suite_name=suite_name)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    token_chunks = [tokenizer.decode([tok]) for tok in token_ids]

    print(f"\n--- Tokenized prompt example | suite={suite_name} style={style} ---")
    print(prompt)
    print("token_ids:")
    print(token_ids)
    print("decoded_token_chunks:")
    print(token_chunks)


def evaluate_style(
    model,
    tokenizer,
    style: str,
    max_new_tokens: int,
    suite_name: str,
    examples: list[Example],
) -> dict:
    results = []
    for ex in examples:
        messages = build_messages(style, ex, suite_name=suite_name)
        answer = generate_answer(model, tokenizer, messages, max_new_tokens=max_new_tokens)
        yn = parse_yes_no_prefix(answer)
        follows_target = yn == ex.expected_answer if yn is not None else False
        results.append(
            {
                "expected": ex.expected_answer,
                "answer": answer,
                "yes_no_prefix": yn,
                "follows_target": follows_target,
            }
        )

    n = len(results)
    yes_no_count = sum(r["yes_no_prefix"] is not None for r in results)
    target_count = sum(r["follows_target"] for r in results)
    metric_label = "context-following rate" if suite_name == "context" else "query-correct rate"
    return {
        "suite": suite_name,
        "style": style,
        "yes_no_start_rate": yes_no_count / n,
        "target_following_rate": target_count / n,
        "target_label": metric_label,
        "results": results,
    }


def print_report(report: dict, show_examples: bool) -> None:
    print(f"\n=== Suite: {report['suite']} | Style: {report['style']} ===")
    print(f"yes/no-start rate     : {report['yes_no_start_rate']:.2%}")
    print(f"{report['target_label']:<22}: {report['target_following_rate']:.2%}")
    if not show_examples:
        return
    print("per-example outputs:")
    for i, row in enumerate(report["results"], start=1):
        print(
            f"  {i:02d}. expected={row['expected']:<3} "
            f"prefix={str(row['yes_no_prefix']):<4} answer={row['answer']!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone prompt/eval sweep for Gemma yes-no QA.\n"
            "Runs 10 context-conditioned + 10 query-only examples and reports rates."
        )
    )
    parser.add_argument("--model_id", default="google/gemma-3-1b-it")
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--styles",
        nargs="+",
        default=["system_then_user", "user_only_directive", "few_shot"],
        choices=["system_then_user", "user_only_directive", "few_shot"],
    )
    parser.add_argument("--show_examples", action="store_true")
    parser.add_argument("--show_tokenized_prompts", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"Loading {args.model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()

    suite_specs = [
        ("context", CONTEXT_EXAMPLES),
        ("query_only", QUERY_ONLY_EXAMPLES),
    ]
    for suite_name, examples in suite_specs:
        if args.show_tokenized_prompts:
            probe_example = examples[0]
            for style in args.styles:
                print_tokenized_prompt_example(
                    tokenizer=tokenizer,
                    style=style,
                    suite_name=suite_name,
                    example=probe_example,
                )

        suite_reports = [
            evaluate_style(
                model,
                tokenizer,
                style,
                max_new_tokens=args.max_new_tokens,
                suite_name=suite_name,
                examples=examples,
            )
            for style in args.styles
        ]
        for report in suite_reports:
            print_report(report, show_examples=args.show_examples)

        best = max(
            suite_reports,
            key=lambda r: (r["target_following_rate"], r["yes_no_start_rate"]),
        )
        print(
            f"\n=== Best style for {suite_name} "
            "(by target-following, tie-break yes/no-start) ==="
        )
        print(best["style"])


if __name__ == "__main__":
    main()
