import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

host = os.getenv('PGVECTOR_HOST')
port = os.getenv('PGVECTOR_PORT')
dbname = os.getenv('PGVECTOR_DATABASE')
user = os.getenv('PGVECTOR_USER')
password = os.getenv('PGVECTOR_PASSWORD')

connection_string = f"host={host} port={port} dbname={dbname} user={user} password={password}"

def install_vector_extension():
    try:
        with psycopg.connect(connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM pg_extension WHERE extname='vector';")
                if cur.fetchone():
                    print("VECTOR extensie is al geïnstalleerd.")
                else:
                    cur.execute("CREATE EXTENSION vector;")
                    print("VECTOR extensie succesvol geïnstalleerd.")
                conn.commit()
    except Exception as e:
        print(f"Er is een fout opgetreden: {e}")

install_vector_extension()
