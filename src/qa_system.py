from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain.docstore.document import Document
from datetime import datetime
import logging
import json
import re
from typing import List, Dict, Any, Optional, Union
from pathlib import Path


class InMemoryHistory(BaseChatMessageHistory):
    def __init__(self):
        self.messages = []

    def add_message(self, message):
        self.messages.append(message)

    def clear(self):
        self.messages = []

    def get_messages(self):
        return self.messages


class QASystem:
    def __init__(self, base_dir: Union[str, Path] = ".") -> None:
        self.base_dir: Path = Path(base_dir)
        self.directories: Dict[str, Path] = {
            "PDF_DIR": Path("pdfs"),
            "EXTRACTED_DIR": Path("extracted_texts"),
            "DATA_DIR": Path("data"),
            "LOGS_DIR": Path("logs"),
            "EXPORTS_DIR": Path("exports"),
            "CHROMA_DIR": Path("data/chroma_db"),
        }

        qa_template = """
        Você é um assistente especializado em análise de documentos.
        Use o contexto fornecido para responder à pergunta de forma detalhada e precisa.

        Contexto relevante dos documentos:
        {context}

        Histórico da conversa:
        {chat_history}

        Pergunta: {question}

        Instruções para resposta:
        1. Analise profundamente o contexto fornecido
        2. Identifique os pontos principais relacionados à pergunta
        3. Forneça uma resposta estruturada e completa
        4. Cite exemplos específicos do texto quando relevante
        5. Indique o nível de certeza da resposta
        6. Mencione as seções/documentos específicos usados

        Resposta (estruturada e detalhada):"""

        self.prompt: PromptTemplate = PromptTemplate(
            input_variables=["context", "question", "chat_history"],
            template=qa_template,
        )

        self.qa_chain: Optional[ConversationalRetrievalChain] = None
        self.vectorstore: Optional[Chroma] = None
        self.message_history: Optional[InMemoryHistory] = None

        self.create_directories()
        self.setup_logging()

    def create_directories(self) -> None:
        for directory in self.directories.values():
            directory.mkdir(parents=True, exist_ok=True)
            logging.info(f"Diretório criado/verificado: {directory}")

    def setup_logging(self) -> None:
        log_file: Path = (
            self.directories["LOGS_DIR"]
            / f'qa_system_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        logging.basicConfig(
            filename=str(log_file),
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logging.info("Sistema de logging inicializado")

    def preprocess_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n\s*\n", "\n\n", text)
        text = re.sub(r"[^\w\s.,!?;:()\[\]{}\-\'\"]+", " ", text)
        text = text.replace('"', '"').replace('"', '"')
        text = re.sub(r"([.,!?;:](?:\"|\')?)", r"\1 ", text)
        return text.strip()

    def load_documents(self) -> List[Document]:
        documents = []
        try:
            for file_path in self.directories["EXTRACTED_DIR"].glob("*.txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    processed_text = self.preprocess_text(text)
                    filename = file_path.name
                    documents.append(
                        Document(
                            page_content=processed_text,
                            metadata={
                                "source": file_path.name,
                                "file_path": str(file_path),
                                "created_at": datetime.now().isoformat(),
                                "chunk_id": f"{filename}_0",
                            },
                        )
                    )
            logging.info(f"Carregados e processados {len(documents)} documentos")
            return documents
        except Exception as e:
            logging.error(f"Erro ao carregar documentos: {str(e)}")
            raise

    def initialize_qa_chain(
        self,
        model_name: str = "mistral:7b-instruct",
        temperature: float = 0.3,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k_documents: int = 6,
    ) -> None:
        try:
            documents = self.load_documents()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            )
            chunks = []
            for i, doc in enumerate(documents):
                doc_chunks = text_splitter.split_documents([doc])
            for j, chunk in enumerate(doc_chunks):
                # Preserva os metadados originais e adiciona informação do chunk
                chunk.metadata = {
                    **doc.metadata,
                    "chunk_id": f"{doc.metadata['source']}_{j}",
                }
                chunks.append(chunk)

            embeddings = OllamaEmbeddings(
                model="nomic-embed-text", base_url="http://localhost:11434"
            )

            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=str(self.directories["CHROMA_DIR"]),
            )

            llm = ChatOllama(
                model=model_name,
                temperature=temperature,
                top_k=10,
                top_p=0.9,
                repeat_penalty=1.1,
                base_url="http://localhost:11434",
            )

            self.message_history = InMemoryHistory()

            retriever = self.vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k": k_documents}
            )

            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                return_source_documents=True,
                verbose=True,
                combine_docs_chain_kwargs={"prompt": self.prompt},
            )

            logging.info(f"Sistema QA inicializado com modelo {model_name}")

        except Exception as e:
            logging.error(f"Erro ao inicializar sistema QA: {str(e)}")
            raise

    def process_query(self, query: str) -> Dict[str, Any]:
        if self.qa_chain is None:
            self.initialize_qa_chain()

        try:
            self.message_history.add_message(HumanMessage(content=query))

            # Prepara o histórico de chat no formato esperado
            chat_history = []
            messages = self.message_history.get_messages()
            for i in range(0, len(messages) - 1, 2):
                if i + 1 < len(messages):  # Verifica se há uma resposta correspondente
                    chat_history.append((messages[i].content, messages[i + 1].content))

            result = self.qa_chain.invoke(
                {"question": query, "chat_history": chat_history}
            )

            answer = result["answer"]
            self.message_history.add_message(AIMessage(content=answer))

            sources = []
            for doc in result.get("source_documents", []):
                source_info = {
                    "content": str(doc.page_content),
                    "source": doc.metadata.get("source", "Sem nome"),
                    "file_path": doc.metadata.get(
                        "file_path", "Caminho não disponível"
                    ),
                    "created_at": doc.metadata.get(
                        "created_at", datetime.now().isoformat()
                    ),
                    "chunk_id": doc.metadata.get("chunk_id", "ID não disponível"),
                }
                # Só adiciona se não for duplicata
                if source_info not in sources:
                    sources.append(source_info)

            return {"question": query, "answer": answer, "sources": sources}

        except Exception as e:
            logging.error(f"Erro ao processar pergunta: {str(e)}")
            raise

    def reindex_documents(self) -> None:
        try:
            logging.info("Iniciando reindexação dos documentos...")

            if self.vectorstore:
                self.vectorstore.persist()
                self.vectorstore = None

            if self.message_history:
                self.message_history.clear()

            self.initialize_qa_chain()

            logging.info("Reindexação concluída com sucesso")
        except Exception as e:
            logging.error(f"Erro durante reindexação: {str(e)}")
            raise

    def process_feedback(
        self, question: str, answer: str, rating: int, comment: str = ""
    ) -> None:
        feedback_data = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "rating": rating,
            "comment": comment,
        }

        feedback_file = self.directories["DATA_DIR"] / "feedback.json"

        try:
            feedbacks = []
            if feedback_file.exists():
                with open(feedback_file, "r", encoding="utf-8") as f:
                    feedbacks = json.load(f)

            feedbacks.append(feedback_data)
            with open(feedback_file, "w", encoding="utf-8") as f:
                json.dump(feedbacks, f, ensure_ascii=False, indent=2)

            logging.info(f"Feedback processado: rating={rating}")

        except Exception as e:
            logging.error(f"Erro ao processar feedback: {str(e)}")
            raise

    def export_chat_history(self, custom_filename: Optional[str] = None) -> str:
        try:
            if custom_filename:
                filepath = self.directories["EXPORTS_DIR"] / custom_filename
            else:
                filepath = (
                    self.directories["EXPORTS_DIR"]
                    / f'chat_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
                )

            if self.message_history:
                messages = self.message_history.get_messages()
                export_data = {
                    "timestamp": datetime.now().isoformat(),
                    "conversations": [
                        {
                            "role": "human" if isinstance(msg, HumanMessage) else "ai",
                            "content": msg.content,
                        }
                        for msg in messages
                    ],
                }

                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)

                logging.info(f"Histórico exportado para: {filepath}")
                return str(filepath)

            return ""

        except Exception as e:
            logging.error(f"Erro ao exportar histórico: {str(e)}")
            raise

    def clear_chat_history(self) -> None:
        if self.message_history:
            self.message_history.clear()
        logging.info("Histórico do chat limpo")

    def get_system_stats(self) -> Dict[str, Any]:
        return {
            "total_documents": len(
                list(self.directories["EXTRACTED_DIR"].glob("*.txt"))
            ),
            "total_conversations": len(self.message_history.get_messages()) // 2
            if self.message_history
            else 0,
            "vectorstore_size": len(self.vectorstore.get()) if self.vectorstore else 0,
            "last_interaction": datetime.now().isoformat(),
        }
