import json

import sys
sys.path.append("../") 

from assertion_generator import AssertionGenerator
generator = AssertionGenerator("../preprocessing/assertion_templates.json")

with open("../data/popqa_filtered_v2_enhanced.jsonl", 'r') as f:
    enhanced_facts = [json.loads(line) for line in f]
test_facts = enhanced_facts[:10]

dimension_categories = {
    'form': ['explicit', 'conditional', 'counterfactual', 'imperative', 'interrogative', 'not_at_issue'],
    'epistemic_stance': ['strong', 'weak'],
    'evidentiality': ['hearsay', 'authority', 'belief_reports'],
    'tone': ['informal', 'poetic', 'child_directed', 'emotional_appeal', 'sarcasm', 'social_media']
}

# Generate small test dataset
samples_per_combination = 2
test_dataset, failures = generator.generate_balanced_dataset(
    test_facts, 
    dimension_categories, 
    samples_per_combination=samples_per_combination
)

print(f"\nTest generation results:")
print(f"Generated {len(test_dataset)} test assertions")
print(f"Failed {len(failures)} generations")

# Show samples
for assertion in test_dataset[:17*samples_per_combination]:
    print(f"\n[{assertion['dimension']}-{assertion['category']}]")
    print(f"Assertion: {assertion['assertion']}")
    print(f"Query: {assertion['query']}")