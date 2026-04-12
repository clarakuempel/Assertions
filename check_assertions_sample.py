#!/usr/bin/env python3
"""
Script to randomly sample 1000 rows from generated_assertions_v2_2000.jsonl
and check if the assertions make syntactic and semantic sense.
"""

import json
import random
from collections import defaultdict

# Set seed for reproducibility
random.seed(42)

# Read all rows
data = []
with open('data/generated_assertions_v2_2000.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line))

print(f"Total rows in file: {len(data)}")

# Randomly sample 1000 rows
sample = random.sample(data, min(1000, len(data)))
print(f"Sampled {len(sample)} rows\n")

# Analyze the assertions
issues = []
categories_checked = defaultdict(int)
dimensions_checked = defaultdict(int)

for i, row in enumerate(sample):
    assertion = row.get('assertion', '')
    category = row.get('category', 'unknown')
    dimension = row.get('dimension', 'unknown')
    
    categories_checked[category] += 1
    dimensions_checked[dimension] += 1
    
    # Check for basic issues
    has_issue = False
    issue_types = []
    
    # Check 1: Empty or too short
    if not assertion or len(assertion.strip()) < 5:
        has_issue = True
        issue_types.append("Empty or too short")
    
    # Check 2: Missing punctuation at end
    if assertion and not assertion.rstrip()[-1] in '.?!':
        has_issue = True
        issue_types.append("Missing end punctuation")
    
    # Check 3: Incomplete sentences (basic check)
    if assertion:
        # Check for common patterns of incomplete sentences
        if assertion.endswith(' is') or assertion.endswith(' are') or assertion.endswith(' the'):
            has_issue = True
            issue_types.append("Incomplete sentence")
        
        # Check for malformed conditionals
        if 'If ' in assertion and ' then ' not in assertion.lower():
            # Could be okay for some formats, but flag for review
            pass
    
    # Check 4: Unusual characters or encoding issues
    if any(ord(c) > 127 and c not in 'áéíóúñü' for c in assertion):
        # Allow common accented characters, but flag others
        pass
    
    if has_issue:
        issues.append({
            'index': i,
            'assertion': assertion,
            'category': category,
            'dimension': dimension,
            'issues': issue_types
        })

# Print summary statistics
print("="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(f"\nTotal assertions checked: {len(sample)}")
print(f"Assertions with potential issues: {len(issues)}")
print(f"Issue rate: {len(issues)/len(sample)*100:.2f}%\n")

print("Categories checked:")
for category, count in sorted(categories_checked.items(), key=lambda x: -x[1]):
    print(f"  {category}: {count}")

print("\nDimensions checked:")
for dimension, count in sorted(dimensions_checked.items(), key=lambda x: -x[1]):
    print(f"  {dimension}: {count}")

# Show some examples from each category and dimension
print("\n" + "="*80)
print("SAMPLE ASSERTIONS BY CATEGORY AND DIMENSION")
print("="*80)

# Group samples by category and dimension
grouped = defaultdict(list)
for row in sample[:200]:  # Look at first 200 of sample
    key = (row.get('category', 'unknown'), row.get('dimension', 'unknown'))
    if len(grouped[key]) < 3:  # Keep up to 3 examples per group
        grouped[key].append(row.get('assertion', ''))

for (category, dimension), assertions in sorted(grouped.items()):
    print(f"\n{category.upper()} / {dimension.upper()}:")
    for assertion in assertions:
        print(f"  - {assertion}")

# Display issues if any
if issues:
    print("\n" + "="*80)
    print("POTENTIAL ISSUES FOUND")
    print("="*80)
    for issue in issues[:20]:  # Show first 20 issues
        print(f"\nIndex: {issue['index']}")
        print(f"Category: {issue['category']} / Dimension: {issue['dimension']}")
        print(f"Assertion: {issue['assertion']}")
        print(f"Issues: {', '.join(issue['issues'])}")

# Now display a diverse sample for manual review
print("\n" + "="*80)
print("RANDOM SAMPLE FOR MANUAL REVIEW (50 examples)")
print("="*80)

manual_review_sample = random.sample(sample, min(50, len(sample)))
for i, row in enumerate(manual_review_sample):
    print(f"\n{i+1}. [{row.get('category', '?')}/{row.get('dimension', '?')}]")
    print(f"   {row.get('assertion', 'N/A')}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)


