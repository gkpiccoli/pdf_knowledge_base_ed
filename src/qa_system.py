from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from colorama import Fore, Style, init
from datetime import datetime
import logging
import json
import os

# Cria diretórios necessários antes de qualquer outra operação
def create_directories():
    """Cria diretórios necessários se não existirem"""
    directories = ['logs', 'data', 'exports']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Cria os diretórios antes de configurar o logging
create_directories()

# Inicializa colorama para cores no terminal
init()

# Configuração do logging
logging.basicConfig(
    filename=os.path.join('logs', f'qa_system_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Cache para respostas frequentes
response_cache = {}

# Template do prompt melhorado
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

class ConversationManager:
    def __init__(self):
        self.conversations = []
        
    def add_interaction(self, question, answer, sources, feedback):
        self.conversations.append({
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'answer': answer,
            'sources': sources,
            'feedback': feedback
        })
    
    def export_conversations(self):
        filename = f'conversations_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.conversations, f, ensure_ascii=False, indent=2)
        return filename

def create_directories():
    """Cria diretórios necessários se não existirem"""
    directories = ['logs', 'data', 'exports']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def print_colored(text, color, end='\n'):
    """Imprime texto colorido"""
    print(f"{color}{text}{Style.RESET_ALL}", end=end)

def get_feedback():
    """Obtém feedback detalhado do usuário"""
    print_colored("\nPor favor, avalie a resposta:", Fore.YELLOW)
    print("1 - Excelente")
    print("2 - Boa")
    print("3 - Regular")
    print("4 - Ruim")
    print("5 - Muito ruim")
    
    while True:
        try:
            rating = int(input("Avaliação (1-5): "))
            if 1 <= rating <= 5:
                return rating
            print("Por favor, digite um número entre 1 e 5")
        except ValueError:
            print("Por favor, digite um número válido")

def load_documents(directory):
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

def create_qa_system():
    try:
        print_colored("Iniciando processamento...", Fore.CYAN)
        logging.info("Iniciando processamento do sistema")

        print_colored("1. Criando chunks dos documentos...", Fore.CYAN)
        documents = load_documents("extracted_texts")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Reduzido para chunks menores
            chunk_overlap=100
        )
        chunks = text_splitter.create_documents(documents)
        print_colored(f"   Criados {len(chunks)} chunks\n", Fore.GREEN)

        print_colored("2. Criando base vetorial...", Fore.CYAN)
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="data/chroma_db"
        )

        print_colored("\n3. Configurando modelo de chat...", Fore.CYAN)
        llm = ChatOllama(
            model="mistral",
            temperature=0.3  # Reduzido para respostas mais precisas
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

def main():
    try:
        create_directories()
        qa_chain = create_qa_system()
        conversation_manager = ConversationManager()
        
        print_colored("\nSistema pronto! Digite suas perguntas (ou comandos especiais):", Fore.GREEN)
        print_colored("Comandos disponíveis:", Fore.YELLOW)
        print("- 'sair': Encerra o sistema")
        print("- 'exportar': Exporta histórico de conversas")
        print("- 'ajuda': Mostra comandos disponíveis")
        
        chat_history = []

        while True:
            try:
                question = input(f"\n{Fore.GREEN}Pergunta: {Style.RESET_ALL}").strip()
                
                if question.lower() == 'sair':
                    break
                elif question.lower() == 'exportar':
                    filename = conversation_manager.export_conversations()
                    print_colored(f"\nConversas exportadas para: {filename}", Fore.GREEN)
                    continue
                elif question.lower() == 'ajuda':
                    print_colored("\nComandos disponíveis:", Fore.YELLOW)
                    print("- 'sair': Encerra o sistema")
                    print("- 'exportar': Exporta histórico de conversas")
                    print("- 'ajuda': Mostra comandos disponíveis")
                    continue
                
                if not question:
                    continue
                
                # Verifica cache
                cache_key = (question, str(chat_history))
                if cache_key in response_cache:
                    print_colored("\n[Resposta do cache]", Fore.YELLOW)
                    result = response_cache[cache_key]
                else:
                    logging.info(f"Pergunta recebida: {question}")
                    result = qa_chain.invoke({
                        "question": question,
                        "chat_history": chat_history
                    })
                    response_cache[cache_key] = result
                
                answer = result["answer"]
                sources = [doc.metadata.get("source", "Fonte não especificada") 
                          for doc in result["source_documents"]]
                
                print_colored("\nResposta:", Fore.BLUE)
                print(answer)
                
                print_colored("\nFontes:", Fore.YELLOW)
                for source in sources:
                    print("-", source)
                
                feedback = get_feedback()
                conversation_manager.add_interaction(question, answer, sources, feedback)
                
                chat_history.append((question, answer))
                logging.info(f"Resposta fornecida com feedback {feedback}")

            except Exception as e:
                logging.error(f"Erro durante a interação: {str(e)}")
                print_colored(f"\nOcorreu um erro: {str(e)}", Fore.RED)
                print("Você pode continuar fazendo perguntas.")

    except Exception as e:
        logging.error(f"Erro fatal: {str(e)}")
        print_colored(f"\nErro fatal: {str(e)}", Fore.RED)
    finally:
        print_colored("\nExportando conversas finais...", Fore.YELLOW)
        conversation_manager.export_conversations()
        print_colored("Sistema encerrado", Fore.RED)
        logging.info("Sistema encerrado")

if __name__ == "__main__":
    main()