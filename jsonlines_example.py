#!/usr/bin/env python3
"""
Example script for using the jsonlines library to write/read JSONL files.
The jsonlines library provides a clean, convenient interface for working with JSONL files.
"""

import jsonlines
from typing import List, Dict, Any


def write_dicts_to_jsonl_with_jsonlines(data: List[Dict[str, Any]], filename: str) -> None:
    """
    Write a list of dictionaries to a JSONL file using the jsonlines library.
    
    Args:
        data: List of dictionaries to write
        filename: Output filename (should end with .jsonl)
    """
    with jsonlines.open(filename, mode='w') as writer:
        writer.write_all(data)
    
    print(f"Successfully wrote {len(data)} records to {filename}")


def write_dicts_one_by_one(data: List[Dict[str, Any]], filename: str) -> None:
    """
    Alternative: Write dictionaries one by one using jsonlines.
    
    Args:
        data: List of dictionaries to write
        filename: Output filename (should end with .jsonl)
    """
    with jsonlines.open(filename, mode='w') as writer:
        for item in data:
            writer.write(item)
    
    print(f"Successfully wrote {len(data)} records to {filename} (one by one)")


def read_jsonl_with_jsonlines(filename: str) -> List[Dict[str, Any]]:
    """
    Read a JSONL file using the jsonlines library.
    
    Args:
        filename: JSONL file to read
        
    Returns:
        List of dictionaries
    """
    data = []
    with jsonlines.open(filename) as reader:
        for obj in reader:
            data.append(obj)
    return data


def read_jsonl_as_list(filename: str) -> List[Dict[str, Any]]:
    """
    Alternative: Read entire JSONL file as a list in one go.
    
    Args:
        filename: JSONL file to read
        
    Returns:
        List of dictionaries
    """
    with jsonlines.open(filename) as reader:
        return list(reader)


def append_to_jsonl(new_data: List[Dict[str, Any]], filename: str) -> None:
    """
    Append new data to an existing JSONL file.
    
    Args:
        new_data: List of dictionaries to append
        filename: Existing JSONL file
    """
    with jsonlines.open(filename, mode='a') as writer:
        writer.write_all(new_data)
    
    print(f"Successfully appended {len(new_data)} records to {filename}")


def main():
    # Example data: list of dictionaries
    sample_data = [
        {
            "id": 1,
            "name": "Alice Johnson",
            "email": "alice@example.com",
            "age": 28,
            "city": "New York",
            "skills": ["Python", "Machine Learning", "Data Analysis"]
        },
        {
            "id": 2,
            "name": "Bob Smith",
            "email": "bob@example.com",
            "age": 34,
            "city": "San Francisco",
            "skills": ["JavaScript", "React", "Node.js"]
        },
        {
            "id": 3,
            "name": "Carol Davis",
            "email": "carol@example.com",
            "age": 29,
            "city": "Chicago",
            "skills": ["SQL", "Tableau", "Statistics"]
        }
    ]
    
    # Method 1: Write all data at once
    print("=== Method 1: write_all() ===")
    output_filename = "jsonlines_output.jsonl"
    write_dicts_to_jsonl_with_jsonlines(sample_data, output_filename)
    
    # Method 2: Write one by one
    print("\n=== Method 2: write() one by one ===")
    output_filename2 = "jsonlines_output2.jsonl"
    write_dicts_one_by_one(sample_data, output_filename2)
    
    # Read the data back
    print("\n=== Reading data back ===")
    loaded_data = read_jsonl_with_jsonlines(output_filename)
    
    for record in loaded_data:
        print(f"- {record['name']} ({record['age']}) from {record['city']}")
        print(f"  Skills: {', '.join(record['skills'])}")
    
    print(f"\nTotal records loaded: {len(loaded_data)}")
    
    # Demonstrate appending
    print("\n=== Appending new data ===")
    new_records = [
        {
            "id": 4,
            "name": "David Wilson",
            "email": "david@example.com",
            "age": 42,
            "city": "Austin",
            "skills": ["Go", "Kubernetes", "DevOps"]
        }
    ]
    
    append_to_jsonl(new_records, output_filename)
    
    # Read the updated file
    print("\nAfter appending:")
    updated_data = read_jsonl_as_list(output_filename)
    print(f"Total records now: {len(updated_data)}")
    
    # Show the last record
    last_record = updated_data[-1]
    print(f"Last record: {last_record['name']} from {last_record['city']}")


if __name__ == "__main__":
    main() 