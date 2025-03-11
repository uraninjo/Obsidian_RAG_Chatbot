import os
import glob
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import AIMessage, HumanMessage
from pathlib import Path
from langchain_community.vectorstores import Chroma
from colorama import Fore, Style

load_dotenv()

def create_vector_store(docs, store_name, embeddings, db_dir):
    persistent_directory = os.path.join(db_dir, store_name)
    if not os.path.exists(persistent_directory):
        print(Fore.YELLOW + f"\n--- Creating vector store {store_name} ---" + Style.RESET_ALL)
        db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
        db.persist()
        print(Fore.GREEN + f"--- Finished creating vector store {store_name} ---" + Style.RESET_ALL)
    else:
        print(Fore.CYAN + f"Vector store {store_name} already exists." + Style.RESET_ALL)

def query_vector_store(store_name, query, db_dir, embeddings, search_type="mmr", search_kwargs={"k": 15, "fetch_k": 50, "lambda_mult": 0.5}):
    persistent_directory = os.path.join(db_dir, store_name)

    if not os.path.exists(persistent_directory):
        print(Fore.RED + f"Vector store {store_name} does not exist." + Style.RESET_ALL)
        return

    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
    retriever = db.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )
    relevant_docs = retriever.invoke(query)
    
    print(Fore.MAGENTA + f"\n--- Relevant Documents for {store_name} ---" + Style.RESET_ALL)
    for i, doc in enumerate(relevant_docs, 1):
        print(Fore.BLUE + f"Document {i}:\n{doc.page_content}\n" + Style.RESET_ALL)
        if doc.metadata:
            print(Fore.CYAN + f"Source: {doc.metadata.get('source', 'Unknown')}\n" + Style.RESET_ALL)
    return relevant_docs

def get_retriever(store_name, db_dir, embeddings, search_type="mmr", search_kwargs={"k": 15, "fetch_k": 50, "lambda_mult": 0.5}):
    persistent_directory = os.path.join(db_dir, store_name)

    if not os.path.exists(persistent_directory):
        print(Fore.RED + f"Vector store {store_name} does not exist." + Style.RESET_ALL)
        return

    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
    retriever = db.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )
    return retriever