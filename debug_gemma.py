from transformers import AutoTokenizer

# Load the tokenizer (Gemma 2 and 3 share the same tokenizer logic/vocabulary)
model_id = "google/gemma-3-1b-it" 
print(f"Loading tokenizer for {model_id}...\n")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Our test strings
text_clean = "Hello world"
text_literal_bos = "<bos>Hello world"

print("--- TEST 1: The 'Literal String' Trap ---")
# We disable automatic special tokens to see what it does with the literal string
tokens_literal = tokenizer(text_literal_bos, add_special_tokens=False)["input_ids"]
decoded_literal = [tokenizer.decode([t]) for t in tokens_literal]

print(f"Input string:  '{text_literal_bos}'")
print(f"Token IDs:     {tokens_literal}")
print(f"Decoded chunks: {decoded_literal}")
if tokens_literal[0] != tokenizer.bos_token_id:
    print("❌ FAILURE: The tokenizer chopped '<bos>' into regular text pieces instead of mapping to ID 2.\n")

print("--- TEST 2: The Automatic Method (Standard) ---")
# We let the tokenizer do its job automatically on the clean string
tokens_auto = tokenizer(text_clean, add_special_tokens=True)["input_ids"]
decoded_auto = [tokenizer.decode([t]) for t in tokens_auto]

print(f"Input string:  '{text_clean}' (with add_special_tokens=True)")
print(f"Token IDs:     {tokens_auto}")
print(f"Decoded chunks: {decoded_auto}")
if tokens_auto[0] == tokenizer.bos_token_id:
    print("✅ SUCCESS: The tokenizer automatically prepended ID 2.\n")

print("--- TEST 3: The Manual Prepend Method (Custom Pipelines) ---")
# We manually inject the BOS token ID (2) to the front of the array
tokens_raw = tokenizer(text_clean, add_special_tokens=False)["input_ids"]
tokens_manual = [tokenizer.bos_token_id] + tokens_raw
decoded_manual = [tokenizer.decode([t]) for t in tokens_manual]

print(f"Input string:  Prepend ID 2 to '{text_clean}'")
print(f"Token IDs:     {tokens_manual}")
print(f"Decoded chunks: {decoded_manual}")
if tokens_manual[0] == tokenizer.bos_token_id:
    print("✅ SUCCESS: The BOS token ID is safely at the front.\n")

print("--- TEST 4: apply_chat_template (Gemma chat formatting) ---")
rounds = [
    {"role": "system", "content": "Answer the question with Yes or No."},
    {"role": "user", "content": "The sky is blue. Is that true?"},
]
chat_text = tokenizer.apply_chat_template(rounds, tokenize=False, add_generation_prompt=True)
chat_tokens = tokenizer(chat_text, add_special_tokens=False)["input_ids"]

print("Rendered chat template:")
print(chat_text)
print(f"Starts with <bos>: {chat_text.startswith(tokenizer.bos_token)}")
print(f"First token is BOS ID: {chat_tokens[0] == tokenizer.bos_token_id}")
print(f"Contains user turn marker: {'<start_of_turn>user' in chat_text}")
print(f"Contains model turn marker: {'<start_of_turn>model' in chat_text}")