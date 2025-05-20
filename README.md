# Assertion Typology Framework

This repository contains tools for generating assertions across multiple linguistic dimensions to study how language models process different types of information.

## Overview

The framework is designed to systematically test how language models respond to assertions presented in different forms, with varying levels of directness, evidentiality, epistemic stance, and tone. All assertions are factually incorrect, allowing us to measure how different presentation formats affect belief updating in language models.

## Files

- `assertion_templates.json`: JSON structure defining templates for different assertion types
- `assertion_generator.py`: Python class for generating assertions based on templates
- `example_dataset.jsonl`: JSONL file with pre-generated assertions across dimensions

## Getting Started

### Installation

```bash
git clone https://github.com/yourusername/assertion-typology.git
cd assertion-typology
pip install -r requirements.txt
```

### Basic Usage

```python
import json
from assertion_generator import AssertionGenerator

generator = AssertionGenerator("assertion_templates.json")
    
# Generate a simple assertion
print("Form (explicit):", generator.generate_assertion("form", "explicit"))

# Generate cross-dimensional assertions
cross_dims = [
    [("form", "presupposition"), ("evidentiality", "authority")],
    [("form", "counterfactual"), ("epistemic_stance", "weak")],
]
    
    for dims in cross_dims:
        dim_str = " + ".join([f"{d[0]} ({d[1]})" for d in dims])
        print(f"{dim_str}:", generator.generate_cross_dimensional_assertion(dims))
    
```

### Generating Datasets

```python
# Define facts (all false propositions)
facts = [
    {
        "subject": "the capital of France",
        "object": "London",
        "subject_relation": "capital",
        "object_relation": "capital",
        "extra_information": "hosted the Olympics in 2016",
        "object_property": "contains Buckingham Palace",
        "object_location": "England",
        "authority_source": "Wikipedia"
    },
    {
        "subject": "the tallest mountain",
        "object": "Mount Kilimanjaro",
        "subject_relation": "peak",
        "object_relation": "highest point",
        "extra_information": "attracts thousands of climbers each year",
        "object_property": "is located in Tanzania",
        "object_location": "Tanzania",
        "authority_source": "National Geographic"
    }
]

# Generate examples varying the form dimension
dataset = generator.generate_dataset(facts, ["form"])

# Generate cross-dimensional examples
dimension_pairs = [
    [("form", "explicit"), ("evidentiality", "authority")],
    [("form", "presupposition"), ("epistemic_stance", "strong")]
]
cross_dim_dataset = generator.generate_cross_dimensional_dataset(facts, dimension_pairs)

# Save dataset to JSONL
with open("generated_dataset.jsonl", "w") as f:
    for item in dataset:
        f.write(json.dumps(item) + "\n")


# Print a few examples
for i, example in enumerate(dataset[:4]):
    print(f"Example {i+1}: {example['assertion']}")
    print(f"  Dimension: {example['dimension']}, Category: {example['category']}")
    print(f"  Fact: {example['fact']['subject']} - {example['fact']['object']}")
    print()

```

## Typology Dimensions

The framework includes the following linguistic dimensions:

1. **Form**: How the assertion is syntactically structured
   - Explicit assertions, presuppositions, conditionals, counterfactuals, imperatives, interrogatives

2. **Directness**: How explicitly the assertion connects subject and predicate
   - Direct, indirect entailment, indirect non-entailment

3. **Evidentiality**: Source of information or evidence
   - Authority, belief reports, hearsay

4. **Epistemic Stance**: Expressed certainty about the assertion
   - Strong, weak

5. **Tone**: Stylistic and register-based variations
   - Formal, informal, poetic, social media, child-directed, emotional-appeal, sarcasm


## Extending the Framework

### Adding New Templates

Add new templates to `assertion_templates.json` following the existing structure:

```json
"new_dimension": {
  "new_category": {
    "templates": [
      "Template with {placeholder}."
    ],
    "examples": [
      {"template": "Template with {placeholder}.", "example": "Template with value."}
    ]
  }
}
```

### Adding New Dimensions

To add a new dimension:

1. Add the dimension and its categories to `assertion_templates.json`
2. Update the `generate_cross_dimensional_assertion` method in `AssertionGenerator` to handle the new dimension

