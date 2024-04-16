import os
import json
from glob import glob
from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_core.documents import Document
from tqdm import tqdm

load_dotenv()

host = os.getenv('PGVECTOR_HOST')
port = os.getenv('PGVECTOR_PORT')
dbname = os.getenv('PGVECTOR_DATABASE')
user = os.getenv('PGVECTOR_USER')
password = os.getenv('PGVECTOR_PASSWORD')

connection_string = f"postgresql+psycopg://{user}:{password}@{host}:{port}/{dbname}"


embeddings = CohereEmbeddings(model='embed-multilingual-v3.0')

def add_documents_from_json(folder_path):
    json_files = glob(os.path.join(folder_path, '**/*.json'), recursive=True)
    pbar = tqdm(total=len(json_files), desc="Verwerken van bestanden")
    for file_path in json_files:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for item in data:
                collection_name = item['metadata']['collection']
                vectorstore = PGVector(
                    embeddings=embeddings,
                    collection_name=collection_name,
                    connection=connection_string,
                    use_jsonb=True,
                )
                doc = Document(page_content=item['content'], metadata=item['metadata'])
                vectorstore.add_documents([doc])
        pbar.update(1)

folder_path = 'data/LLAMAPARSE/Owners_Manual_modelY_seperatepagechunkjsons'
# data/CLAUDEVISION/Owners_Manual_modelY_seperatepagechunkjsons
# data/LLAMAPARSE/Owners_Manual_modelY_seperatepagechunkjsons
# data/LLAMAPARSE/Owners_Manual_modelY_seperateitemjsons/table_including_summary

add_documents_from_json(folder_path)
