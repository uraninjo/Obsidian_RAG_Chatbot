import os
import glob
import pickle as pkl
import platform
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import AIMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import create_history_aware_retriever
from utils import create_vector_store, get_retriever, search_with_fallback
from langchain_core.runnables import RunnablePassthrough
from colorama import Fore, Style, init

# Terminal renklerini başlat
init(autoreset=True)

# Ortam değişkenlerini yükle
load_dotenv()

# Gemini modeli tanımla
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# İşletim sistemine göre vektör veritabanı yolunu ayarla
if platform.system() == "Windows":
    base_path = Path.home() / "AppData" / "Local" / "Obsidian_rag_db"
else:
    base_path = Path.home() / ".obsidian_rag_db"

db_dir = base_path / "db"
base_path.mkdir(parents=True, exist_ok=True)
db_dir.mkdir(parents=True, exist_ok=True)

pickle_path = base_path / "obsidian_vault.pkl"

# Obsidian vault yolunu yükle veya kullanıcıdan al
if os.path.exists(pickle_path):
    with open(pickle_path, "rb") as f:
        vault_path = pkl.load(f)
else:
    vault_path = input(Fore.YELLOW + "Enter your Obsidian vault path: ")
    with open(pickle_path, "wb") as f:
        pkl.dump(vault_path, f)

# Markdown dosyalarını yükle ve parçalara ayır
md_files = glob.glob(vault_path + "/**/*.md", recursive=True)
documents = []

for file_path in md_files:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text(content)
    for chunk in chunks:
        documents.append(Document(page_content=chunk, metadata={"source": file_path}))

# Embedding modeli tanımla
embedder = SentenceTransformerEmbeddings(model_name='paraphrase-multilingual-MiniLM-L12-v2')

# Vektör veritabanını oluştur
create_vector_store(documents, "chroma_db_with_metadata", embedder, db_dir)

# Geçmiş bazlı sorgu genişletme için prompt
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and the latest user question, reformulate it as a standalone question."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Retriever'ı oluştur
retriever = get_retriever("chroma_db_with_metadata", db_dir, embedder, search_type="mmr", search_kwargs={"k": 10})

history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Keep the answer concise."
    "\n\n"
    "Context: {context}"
)

# QA prompt
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("context"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Alternatif sorgu üretme promptu
query_expansion_prompt = ChatPromptTemplate.from_messages([
    ("system", "If the provided query does not return enough relevant data, generate a better search query."),
    ("human", "{input}"),
])

# Güncellenmiş RAG zinciri
rag_chain = (
    RunnablePassthrough.assign(
        context=lambda x: search_with_fallback(x["input"], retriever, llm, query_expansion_prompt, debug=True)
    )
    | qa_prompt
    | llm
)

if __name__ == "__main__":
    chat_history = []
    while True:
        query = input(Fore.GREEN + "You: ")
        if query.lower() == "exit":
            break
        
        print(Fore.YELLOW + "[INFO] Query işleniyor...")
        response = rag_chain.invoke({"input": query, "chat_history": chat_history})
        
        print(Fore.CYAN + f"AI: {response.content}")
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=response.content))
