import json
import os
import uuid

def save_pages_as_markdown(data, markdown_output_directory):
    for entry in data:
        for page in entry['pages']:
            page_number = page['page']
            page_md_content = page['md']
            md_filename = f'Owners_Manual_modelY_page_{page_number}.md'
            with open(os.path.join(markdown_output_directory, md_filename), 'w') as md_file:
                md_file.write(page_md_content)

def save_items_with_metadata(data, items_output_directory, collection, embeddings_model, max_chunk_size):
    for entry in data:
        for page in entry['pages']:
            page_number = page['page']
            headers = {}
            for item_index, item in enumerate(page['items']):
                item_type = item['type']
                if item_type == 'header':
                    headers[item_index] = item['value']
                if item_type == 'text':
                    item_directory = os.path.join(items_output_directory, 'text')
                    if not os.path.exists(item_directory):
                        os.makedirs(item_directory)
                    chunks = split_text_into_chunks(item['md'], max_chunk_size)
                    for chunk_index, chunk in enumerate(chunks):
                        item_data = {
                            'content': chunk,
                            'metadata': {
                                'id': str(uuid.uuid4()),
                                'source': 'Tesla Usermanual',
                                'page': page_number,
                                'collection': collection,
                                'embeddings_model': embeddings_model,
                                'characters': len(chunk)
                            }
                        }
                        header_index = max(i for i in headers if i < item_index)
                        if header_index in headers:
                            item_data['metadata'][headers[header_index]] = headers[header_index]
                        item_filename = f'text_page_{page_number}_item_{item_index + 1}_chunk_{chunk_index + 1}.json'
                        with open(os.path.join(item_directory, item_filename), 'w') as item_file:
                            json.dump(item_data, item_file, indent=4)

def split_text_into_chunks(text, max_chunk_size):
    words = text.split()
    chunks = []
    current_chunk = []
    current_chunk_size = 0
    for word in words:
        if current_chunk_size + len(word) + 1 <= max_chunk_size:
            current_chunk.append(word)
            current_chunk_size += len(word) + 1
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_chunk_size = len(word) + 1
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(script_dir, '..', '..', '..', 'data', 'LLAMAPARSE')
    json_file_path = os.path.join(base_dir, 'result.json')
    markdown_output_directory = os.path.join(base_dir, 'Owners_Manual_modelY_seperatepagemarkdowns')
    items_output_directory = os.path.join(base_dir, 'Owners_Manual_modelY_seperateitemjsons')
    collection = 'ClaudeVision'
    embeddings_model = 'Cohere'
    max_chunk_size = 500

    with open(json_file_path, 'r') as file:
        data = json.load(file)

    save_items_with_metadata(data, items_output_directory, collection, embeddings_model, max_chunk_size)