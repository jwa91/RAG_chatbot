import json
import uuid
import re
import os
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

def read_prompt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

def generate_table_summary(markdown_table, prompt):
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": markdown_table}
        ]
    )
    # Use attribute access as per the latest API documentation
    return response.choices[0].message.content

def restructure_json(input_file_path, output_file_path, prompt):
    with open(input_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    table_summary = generate_table_summary(data['md'], prompt)
    
    new_data = [{
        "content": table_summary,
        "metadata": {
            "id": str(uuid.uuid4()),
            "source": "Owners_Manual_modelY",
            "page": int(re.search(r"table_page_(\d+)_item", input_file_path).group(1)),
            "collection": "Tables",
            "embeddings_model": "Cohere",
            "characters": data['character_count'],
            "markdown": data['md']
        }
    }]
    
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(new_data, file, indent=4, ensure_ascii=False)

def process_directory(input_dir, output_dir, prompt_path):
    prompt = read_prompt_file(prompt_path)
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    for file in tqdm(json_files, desc="Processing JSON files"):
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, file)
        restructure_json(input_path, output_path, prompt)

prompt_file_path = 'Creating_a_vectorstore/01_DATAPREP/01b_DATAPREP_WITH_LLAMAPARSE/table_summary_prompt.txt'
input_directory = 'data/LLAMAPARSE/Owners_Manual_modelY_seperateitemjsons/table'
output_directory = 'data/LLAMAPARSE/Owners_Manual_modelY_seperateitemjsons/table_including_summary'

process_directory(input_directory, output_directory, prompt_file_path)
