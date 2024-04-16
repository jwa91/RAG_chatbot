import re
import json
from dataclasses import asdict, dataclass
import os
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
import uuid

@dataclass
class Document:
    content: str
    metadata: dict

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

def generate_documents(content, source_name, collection, embeddings_model, max_chunk_size):
    # Split markdown by headers
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(content)
    
    # Split each header group into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=max_chunk_size, chunk_overlap=30)
    documents = []
    for split in md_header_splits:
        chunks = text_splitter.split_text(split.page_content)
        for chunk in chunks:
            metadata = create_metadata(source_name, collection, embeddings_model, {'content': chunk})
            metadata.update(split.metadata)
            documents.append(Document(content=chunk, metadata=metadata))
    return documents

def create_metadata(source_name, collection, embeddings_model, chunk):
    # Extract the page number from the source_name using regular expression
    match = re.search(r'_page_(\d+)', source_name)
    page = int(match.group(1)) if match else 1

    metadata = {
        "id": str(uuid.uuid4()),
        "source": source_name,
        "page": page,
        "collection": collection,
        "embeddings_model": embeddings_model,
        "characters": len(chunk['content'])
    }
    return metadata

def create_documents_from_markdown_file(filename, collection, embeddings_model, max_chunk_size=500):
    with open(filename, 'r') as f:
        content = f.read()
    return generate_documents(content, filename, collection, embeddings_model, max_chunk_size)

def save_documents_to_json(documents, output_directory, filename):
    os.makedirs(output_directory, exist_ok=True)
    output_path = os.path.join(output_directory, filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump([asdict(doc) for doc in documents], f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    markdown_directory = 'data/CLAUDEVISION/Owners_Manual_modelY_seperatepagemarkdowns'
    output_directory = 'data/CLAUDEVISION/Owners_Manual_modelY_seperatepagechunkjsons'
    source_name = 'Tesla Usermanual'
    collection = 'ClaudeVision'
    embeddings_model = 'Cohere'
    max_chunk_size = 500

    if os.path.isdir(markdown_directory):
        for filename in os.listdir(markdown_directory):
            if filename.endswith('.md'):
                markdown_file = os.path.join(markdown_directory, filename)
                output_filename = f"{os.path.splitext(filename)[0]}.json"
                docs = create_documents_from_markdown_file(markdown_file, collection, embeddings_model, max_chunk_size)
                save_documents_to_json(docs, output_directory, output_filename)
    else:
        print(f"De map '{markdown_directory}' bestaat niet. Controleer de locatie en probeer het opnieuw.")

