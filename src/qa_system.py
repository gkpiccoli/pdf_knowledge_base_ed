from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from datetime import datetime
import logging
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path

# Template do prompt
TEMPLATE = """
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

class QASystem:
    def __init__(self, base_dir: Union[str, Path] = ".") -> None:
        """
        Inicializa o sistema de QA.
        
        Args:
            base_dir: Diretório base para armazenamento de arquivos
        """
        self.base_dir: Path = Path(base_dir)
        self.directories: Dict[str, Path] = {
            'logs': self.base_dir / 'logs',
            'data': self.base_dir / 'data',
            'exports': self.base_dir / 'exports',
            'extracted_texts': self.base_dir / 'extracted_texts',
            'PDF': self.base_dir / 'PDF',
            'chroma_db': self.base_dir / 'data' / 'chroma_db'
        }
        self.prompt: PromptTemplate = PromptTemplate(
            input_variables=["context", "question", "chat_history"],
            template=TEMPLATE
        )
        self.qa_chain: Optional[ConversationalRetrievalChain] = None
        self.chat_history: List[Tuple[str, str]] = []
        self.vectorstore: Optional[Chroma] = None
        
        self.create_directories()
        self.setup_logging()

    def create_directories(self) -> None:
        """Cria todos os diretórios necessários para o sistema"""
        for directory in self.directories.values():
            directory.mkdir(parents=True, exist_ok=True)
            logging.info(f"Diretório criado/verificado: {directory}")

    def setup_logging(self) -> None:
        """Configura o sistema de logging com rotação de arquivos"""
        log_file: Path = self.directories['logs'] / f'qa_system_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        logging.basicConfig(
            filename=str(log_file),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logging.info("Sistema de logging inicializado")

    def load_documents(self) -> List[str]:
        """
        Carrega documentos do diretório de textos extraídos.
        
        Returns:
            Lista de documentos carregados
        """
        documents: List[str] = []
        try:
            for file_path in self.directories['extracted_texts'].glob('*.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    documents.append(f.read())
            logging.info(f"Carregados {len(documents)} documentos")
            return documents
        except Exception as e:
            logging.error(f"Erro ao carregar documentos: {str(e)}")
            raise

    def initialize_qa_chain(self, model_name: str = "mistral:7b-instruct", temperature: float = 0.3) -> None:
        """
        Inicializa a chain de QA com configurações específicas.
        
        Args:
            model_name: Nome do modelo a ser usado
            temperature: Temperatura para geração de respostas
        """
        try:
            documents: List[str] = self.load_documents()
            text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100
            )
            chunks: List[Any] = text_splitter.create_documents(documents)

            embeddings: OllamaEmbeddings = OllamaEmbeddings(
                model="nomic-embed-text",
                base_url="http://localhost:11434"
            )
            
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=str(self.directories['chroma_db'])
            )

            llm: ChatOllama = ChatOllama(
                model=model_name,
                temperature=temperature,
                base_url="http://localhost:11434"
            )

            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 4}),
                return_source_documents=True,
                verbose=False,
                combine_docs_chain_kwargs={"prompt": self.prompt}
            )

            logging.info(f"Sistema QA inicializado com modelo {model_name}")

        except Exception as e:
            logging.error(f"Erro ao inicializar sistema QA: {str(e)}")
            raise

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Processa uma pergunta e retorna a resposta com fontes.
        
        Args:
            query: Pergunta do usuário
            
        Returns:
            Dicionário com pergunta, resposta e fontes
        """
        if self.qa_chain is None:
            self.initialize_qa_chain()

        try:
            result: Dict[str, Any] = self.qa_chain.invoke({
                "question": query,
                "chat_history": self.chat_history
            })
            
            answer: str = result["answer"]
            self.chat_history.append((query, answer))
            
            sources: List[str] = [
                str(doc.page_content) 
                for doc in result.get("source_documents", [])
            ]
            
            return {
                "question": query,
                "answer": answer,
                "sources": sources
            }

        except Exception as e:
            logging.error(f"Erro ao processar pergunta: {str(e)}")
            raise

    def process_feedback(self, question: str, answer: str, rating: int, comment: str = "") -> None:
        """
        Processa e salva o feedback do usuário.
        
        Args:
            question: Pergunta original
            answer: Resposta fornecida
            rating: Avaliação numérica
            comment: Comentário opcional
        """
        feedback_data: Dict[str, Any] = {
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'answer': answer,
            'rating': rating,
            'comment': comment
        }
        
        feedback_file: Path = self.directories['data'] / 'feedback.json'
        
        try:
            feedbacks: List[Dict[str, Any]] = []
            if feedback_file.exists():
                with open(feedback_file, 'r', encoding='utf-8') as f:
                    feedbacks = json.load(f)
            
            feedbacks.append(feedback_data)
            with open(feedback_file, 'w', encoding='utf-8') as f:
                json.dump(feedbacks, f, ensure_ascii=False, indent=2)
            
            logging.info(f"Feedback processado: rating={rating}")

        except Exception as e:
            logging.error(f"Erro ao processar feedback: {str(e)}")
            raise

    def export_chat_history(self, custom_filename: Optional[str] = None) -> str:
        """
        Exporta o histórico do chat para um arquivo JSON.
        
        Args:
            custom_filename: Nome personalizado para o arquivo
            
        Returns:
            Caminho do arquivo exportado
        """
        try:
            if custom_filename:
                filepath: Path = self.directories['exports'] / custom_filename
            else:
                filepath: Path = self.directories['exports'] / f'chat_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            
            export_data: Dict[str, Any] = {
                'timestamp': datetime.now().isoformat(),
                'conversations': [
                    {'question': q, 'answer': a} for q, a in self.chat_history
                ]
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            logging.info(f"Histórico exportado para: {filepath}")
            return str(filepath)

        except Exception as e:
            logging.error(f"Erro ao exportar histórico: {str(e)}")
            raise

    def clear_chat_history(self) -> None:
        """Limpa o histórico do chat"""
        self.chat_history = []
        logging.info("Histórico do chat limpo")

    def get_system_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas do sistema.
        
        Returns:
            Dicionário com estatísticas do sistema
        """
        total_docs: int = len(list(self.directories['extracted_texts'].glob('*.txt')))
        total_convs: int = len(self.chat_history)
        vectorstore_size: int = len(self.vectorstore.get()) if self.vectorstore else 0
        
        return {
            "total_documents": total_docs,
            "total_conversations": total_convs,
            "vectorstore_size": vectorstore_size,
            "last_interaction": datetime.now().isoformat()
        }