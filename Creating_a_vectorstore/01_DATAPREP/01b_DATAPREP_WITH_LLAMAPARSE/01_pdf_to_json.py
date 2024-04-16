import os
from llama_parse import LlamaParse
from dotenv import load_dotenv
import json

load_dotenv()

output_directory = 'data/LLAMAPARSE'
images_directory = os.path.join(output_directory, 'images')
os.makedirs(images_directory, exist_ok=True)

parser = LlamaParse(result_type="json", language="nl", verbose=True, max_timeout=20000)
document_path = "tesla/data/Owners_Manual_modelY.pdf"

json_result = parser.get_json_result(document_path)

# Sla het JSON-resultaat op in de output directory
json_output_path = os.path.join(output_directory, 'result.json')
with open(json_output_path, 'w') as f:
    json.dump(json_result, f, indent=2)

# Download de afbeeldingen en sla ze op in de submap 'images'
images = parser.get_images(json_result, images_directory)

print(f"JSON-resultaat opgeslagen in: {json_output_path}")
print(f"Afbeeldingen gedownload naar: {images_directory}")
