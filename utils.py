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

def search_with_fallback(query, retriever, llm, query_expansion_prompt, debug=False):
    if debug:
        print(f"\n{Fore.YELLOW}[DEBUG] Gelen sorgu: {query}{Style.RESET_ALL}")

    # 1️⃣ İlk arama denemesi
    results = retriever.invoke(query)
    if debug:
        print(f"{Fore.YELLOW}[DEBUG] İlk sorgu sonucu: {len(results)} doküman bulundu.{Style.RESET_ALL}")

    if len(results) > 2:  # Eğer yeterli bilgi bulduysa devam et
        results_ = [doc.page_content for doc in results]
        print(f"{Fore.YELLOW}[DEBUG] Dokümanlar: {results_}{Style.RESET_ALL}")
        return results_

    print("🔍 İlk sorgudan yeterli bilgi bulunamadı, sorgu genişletiliyor...")

    # 2️⃣ Alternatif genişletilmiş sorgu oluştur
    new_query = llm.invoke(query_expansion_prompt.format(input=query)).content
    if debug:
        print(f"{Fore.YELLOW}[DEBUG] Yeni oluşturulan sorgu: {new_query}{Style.RESET_ALL}")

    # 3️⃣ Yeni sorgu ile tekrar arama yap
    new_results = retriever.invoke(new_query)
    if debug:
        print(f"{Fore.YELLOW}[DEBUG] Yeni sorgu sonucu: {len(new_results)} doküman bulundu.{Style.RESET_ALL}")

    if len(new_results) > 2:  # Eğer bu sefer sonuç bulursa devam et
        return [doc.page_content for doc in new_results]

    print("⚠️ Genişletilmiş sorgudan da yeterli bilgi bulunamadı. Alternatif çözüm aranıyor...")

    # 4️⃣ Son çare: Kullanıcıya bilgi yok mesajı göster
    return ["I couldn't find enough relevant information. Maybe try rephrasing your question?"]
