from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from datetime import datetime
import logging
import json
import os

# Template do prompt
template = """
Você é um assistente especializado em responder perguntas com base nos documentos fornecidos.
Seu objetivo é fornecer respostas precisas, claras e bem fundamentadas.

Contexto dos documentos:
{context}

Histórico da conversa:
{chat_history}

Pergunta atual: {question}

Instruções específicas:
1. Responda de forma direta e objetiva
2. Cite as fontes relevantes
3. Se houver incerteza, explique o nível de confiança
4. Mantenha o foco no contexto fornecido
5. Use exemplos quando apropriado

Resposta:"""

PROMPT = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template=template
)

def create_directories():
    """Cria diretórios necessários se não existirem"""
    directories = ['logs', 'data', 'exports', 'extracted_texts']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def setup_logging():
    """Configura o sistema de logging"""
    logging.basicConfig(
        filename=os.path.join('logs', f'qa_system_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def load_documents(directory):
    """Carrega documentos do diretório especificado"""
    documents = []
    try:
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                    documents.append(f.read())
        logging.info(f"Carregados {len(documents)} documentos")
    except Exception as e:
        logging.error(f"Erro ao carregar documentos: {str(e)}")
        raise
    return documents

def setup_qa_chain():
    """Configura e retorna a chain de QA"""
    try:
        # Carrega e processa documentos
        documents = load_documents("extracted_texts")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        chunks = text_splitter.create_documents(documents)

        # Configura embeddings e vectorstore
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="data/chroma_db"
        )

        # Configura o modelo e a chain
        llm = ChatOllama(
            model="mistral",
            temperature=0.3
        )

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True,
            verbose=False,
            combine_docs_chain_kwargs={"prompt": PROMPT}
        )

        return qa_chain

    except Exception as e:
        logging.error(f"Erro ao criar sistema: {str(e)}")
        raise

def process_feedback(question: str, answer: str, rating: int):
    """Processa e salva o feedback do usuário"""
    feedback_data = {
        'timestamp': datetime.now().isoformat(),
        'question': question,
        'answer': answer,
        'rating': rating
    }
    
    feedback_file = os.path.join('data', 'feedback.json')
    
    # Carrega feedbacks existentes ou cria lista vazia
    if os.path.exists(feedback_file):
        with open(feedback_file, 'r', encoding='utf-8') as f:
            feedbacks = json.load(f)
    else:
        feedbacks = []
    
    # Adiciona novo feedback e salva
    feedbacks.append(feedback_data)
    with open(feedback_file, 'w', encoding='utf-8') as f:
        json.dump(feedbacks, f, ensure_ascii=False, indent=2)

def export_chat_history(chat_history, filepath):
    """Exporta o histórico do chat para um arquivo"""
    export_data = {
        'timestamp': datetime.now().isoformat(),
        'conversations': [
            {'question': q, 'answer': a} for q, a in chat_history
        ]
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)