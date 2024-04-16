import os
from uuid import uuid4
from dotenv import load_dotenv
from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_postgres.vectorstores import PGVector
from langchain_cohere import CohereEmbeddings

load_dotenv()

# Set up LangSmith tracing
unique_id = uuid4().hex[:8]
os.environ["LANGCHAIN_PROJECT"] = f"Teslachatbot- {unique_id}"
client = Client(api_key=os.environ["LANGSMITH_API_KEY"])

def initialize_db_connection():
    host = os.getenv('PGVECTOR_HOST')
    port = os.getenv('PGVECTOR_PORT')
    dbname = os.getenv('PGVECTOR_DATABASE')
    user = os.getenv('PGVECTOR_USER')
    password = os.getenv('PGVECTOR_PASSWORD')
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{dbname}"

def initialize_vectorstore(connection_string, collection_name):
    embeddings = CohereEmbeddings(model='embed-multilingual-v3.0')
    return PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection_string,
        use_jsonb=True
    )


def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever(search_type="mmr")
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query in Dutch to look up in order to get information relevant to the conversation")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input, vector_store, chat_history):
    retriever_chain = get_context_retriever_chain(vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    response = conversation_rag_chain.invoke({
        "chat_history": chat_history,
        "input": user_input
    })
    return response['answer']

def main():
    connection_string = initialize_db_connection()
    vector_store = initialize_vectorstore(connection_string, "Llamaparse")
    chat_history = [AIMessage(content="Hello, I am a bot. How can I help you?")]

    while True:
        user_query = input("Type your message here (type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
        response = get_response(user_query, vector_store, chat_history)
        chat_history.append(HumanMessage(content=user_query))
        chat_history.append(AIMessage(content=response))
        print(f"AI: {response}")

if __name__ == "__main__":
    main()
