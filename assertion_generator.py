import json
import random
import itertools
from typing import Dict, List, Any, Tuple

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
        
        return result
    
    def generate_assertion(self, dimension: str, category: str, placeholders: Dict[str, str] = None) -> str:
        """Generate an assertion for a specific dimension and category."""
        template = self.get_random_template(dimension, category)
        return self.fill_template(template, placeholders)
    
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
                    
                    fact_examples.append({
                        "fact": fact,
                        "dimension": dimension,
                        "category": category,
                        "assertion": assertion
                    })
            
            dataset.extend(fact_examples)
            
        return dataset
    
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


# Example usage
if __name__ == "__main__":
    generator = AssertionGenerator("assertion-templates.json")
    
    # Generate a simple assertion
    print("SINGLE DIMENSION EXAMPLES:")
    print("Form (explicit):", generator.generate_assertion("form", "explicit"))
    print("Form (presupposition):", generator.generate_assertion("form", "presupposition"))
    print("Directness (indirect_entailment):", generator.generate_assertion("directness", "indirect_entailment"))
    print("Evidentiality (authority):", generator.generate_assertion("evidentiality", "authority"))
    print("Epistemic stance (weak):", generator.generate_assertion("epistemic_stance", "weak"))
    print("Tone (formal):", generator.generate_assertion("tone", "formal"))
    
    print("\nCROSS-DIMENSIONAL EXAMPLES:")
    # Generate cross-dimensional assertions
    cross_dims = [
        [("form", "presupposition"), ("evidentiality", "authority")],
        [("form", "counterfactual"), ("epistemic_stance", "weak")],
        [("form", "explicit"), ("tone", "formal")],
        [("form", "interrogative"), ("evidentiality", "hearsay")]
    ]
    
    for dims in cross_dims:
        dim_str = " + ".join([f"{d[0]} ({d[1]})" for d in dims])
        print(f"{dim_str}:", generator.generate_cross_dimensional_assertion(dims))
    
    print("\nDATASET GENERATION EXAMPLE:")
    # Generate a dataset with different facts
    facts = [
        {
            "subject": "the capital of France",
            "object": "London",
            "subject_relation": "capital",
            "object_relation": "capital"
        },
        {
            "subject": "the tallest mountain",
            "object": "Mount Kilimanjaro",
            "subject_relation": "peak",
            "object_relation": "highest point"
        }
    ]
    
    # Generate examples varying the form dimension
    dataset = generator.generate_dataset(facts, ["form"])
    
    # Print a few examples
    for i, example in enumerate(dataset[:4]):
        print(f"Example {i+1}: {example['assertion']}")
        print(f"  Dimension: {example['dimension']}, Category: {example['category']}")
        print(f"  Fact: {example['fact']['subject']} - {example['fact']['object']}")
        print()