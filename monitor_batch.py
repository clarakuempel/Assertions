from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
client = OpenAI()

batches = client.batches.list(limit=10)
print(batches)
# print(client.batches.retrieve(batches.data[0].id))
# file_response = client.files.content(batches.data[0].input_file_id)
# print(file_response.text)