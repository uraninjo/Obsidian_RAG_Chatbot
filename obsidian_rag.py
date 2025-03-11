import os
import glob
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from colorama import Fore, Style, init
import platform
import pickle as pkl
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from utils import continual_chat, create_vector_store, get_retriever

# Load environment variables from .env
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# İşletim sistemini belirleme
if platform.system() == "Windows":
    base_path = Path.home() / "AppData" / "Local" / "Obsidian_rag_db"
else:  # Linux (Ubuntu) için
    base_path = Path.home() / ".obsidian_rag_db"

# Klasörü oluşturma
db_dir = base_path / "db"
base_path.mkdir(parents=True, exist_ok=True)
db_dir.mkdir(parents=True, exist_ok=True)

pickle_path = base_path / "obisidan_vault.pkl"

if os.path.exists(pickle_path):
    with open(pickle_path, "rb") as f:
        vault_path = pkl.load(f)
    if vault_path == "":
        vault_path = input(Fore.YELLOW + "Enter your Obsidian vault path: " + Style.RESET_ALL)
        with open(pickle_path, "wb") as f:
            pkl.dump(vault_path, f)
        print(Fore.CYAN + f"Vault path saved to {pickle_path}" + Style.RESET_ALL)
    else:
        print(Fore.GREEN + f"Vault path loaded from {pickle_path}" + Style.RESET_ALL)
else:
    vault_path = input(Fore.YELLOW + "Enter your Obsidian vault path: " + Style.RESET_ALL)  #"/mnt/d/Obsidian/DesktopUra"
    with open(pickle_path, "wb") as f:
        pkl.dump(vault_path, f)
    print(Fore.CYAN + f"Vault path saved to {pickle_path}" + Style.RESET_ALL)

# Load and process Markdown documents
md_files = glob.glob(vault_path + "/**/*.md", recursive=True)
documents = []

for file_path in md_files:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text(content)
    for chunk in chunks:
        documents.append(Document(page_content=chunk, metadata={"source": file_path}))

print(Fore.MAGENTA + f"{len(documents)} documents processed." + Style.RESET_ALL)

# Initialize embeddings
embedder = SentenceTransformerEmbeddings(model_name='paraphrase-multilingual-MiniLM-L12-v2')

# Create and save vector store
create_vector_store(documents, "chroma_db_with_metadata", embedder, db_dir)

# Contextualize question prompt
# This system prompt helps the AI understand that it should reformulate the question
# based on the chat history to make it a standalone question
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

# Create a prompt template for contextualizing questions
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

retriever = get_retriever("chroma_db_with_metadata", db_dir, embedder, search_type="mmr", search_kwargs={"k": 3, "fetch_k": 20, "lambda_mult": 0.5})

# Create a history-aware retriever
# This uses the LLM to help reformulate the question based on chat history
history_aware_retriever = create_history_aware_retriever(
    model, retriever, contextualize_q_prompt
)

# Answer question prompt
# This system prompt helps the AI understand that it should provide concise answers
# based on the retrieved context and indicates what to do if the answer is unknown
qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer concise."
    "\n\n"
    "{context}"
)

# Create a prompt template for answering questions
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a chain to combine documents for question answering
# `create_stuff_documents_chain` feeds all retrieved context into the LLM
question_answer_chain = create_stuff_documents_chain(model, qa_prompt)

# Create a retrieval chain that combines the history-aware retriever and the question answering chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

if __name__ == "__main__":
    continual_chat(rag_chain)

