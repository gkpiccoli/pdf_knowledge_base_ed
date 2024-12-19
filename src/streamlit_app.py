import streamlit as st
import os
from datetime import datetime
import time
from qa_system import create_directories, setup_qa_chain, process_feedback, export_chat_history

# Desabilitar PostHog
os.environ['POSTHOG_API_KEY'] = ''

# Configuração da página
st.set_page_config(
    page_title="PDF Knowledge Base QA",
    page_icon="📚",
    layout="wide"
)

# Estilo CSS personalizado
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextInput > div > div > input {
        padding: 0.5rem;
    }
    .feedback-button {
        margin: 0.2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Função para inicializar o sistema com feedback visual
def initialize_system():
    try:
        with st.spinner('Criando diretórios necessários...'):
            create_directories()
            time.sleep(1)
        
        init_message = st.empty()
        init_message.info('Inicializando o sistema de QA... (isso pode levar alguns minutos)')
        
        # Inicializa o QA chain
        qa_chain = setup_qa_chain()
        
        init_message.success('Sistema inicializado com sucesso!')
        time.sleep(2)
        init_message.empty()
        
        return qa_chain
    except Exception as e:
        st.error(f"Erro ao inicializar o sistema: {str(e)}")
        st.error("Verifique se o Ollama está rodando e se os documentos estão no diretório correto.")
        return None

# Inicialização das variáveis de sessão
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False
    st.session_state.chat_history = []

# Título principal
st.title("📚 Sistema de Perguntas e Respostas")

# Inicialização do sistema
if not st.session_state.system_initialized:
    qa_chain = initialize_system()
    if qa_chain is not None:
        st.session_state.qa_chain = qa_chain
        st.session_state.system_initialized = True
        st.rerun()

# Se o sistema está inicializado, mostra a interface principal
if st.session_state.system_initialized:
    # Sidebar com informações
    with st.sidebar:
        st.header("Sobre")
        st.info("""
        Este sistema permite fazer perguntas sobre os documentos carregados.
        As respostas são geradas com base no conteúdo dos documentos processados.
        """)
        
        st.header("Estatísticas")
        if st.session_state.chat_history:
            st.metric("Perguntas Realizadas", len(st.session_state.chat_history))
        
        if st.button("Exportar Histórico"):
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                export_path = os.path.join("exports", f"chat_history_{timestamp}.json")
                export_chat_history(st.session_state.chat_history, export_path)
                st.success(f"Histórico exportado para {export_path}")
            except Exception as e:
                st.error(f"Erro ao exportar histórico: {str(e)}")

    # Área principal de chat
    st.header("💬 Chat")

    # Exibir histórico de mensagens
    for i, (question, answer) in enumerate(st.session_state.chat_history):
        st.info(f"Pergunta: {question}")
        st.success(f"Resposta: {answer}")
        
        # Botões de feedback
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            if st.button("⭐", key=f"fb1_{i}"):
                process_feedback(question, answer, 1)
                st.success("Feedback registrado!")
        with col2:
            if st.button("⭐⭐", key=f"fb2_{i}"):
                process_feedback(question, answer, 2)
                st.success("Feedback registrado!")
        with col3:
            if st.button("⭐⭐⭐", key=f"fb3_{i}"):
                process_feedback(question, answer, 3)
                st.success("Feedback registrado!")
        with col4:
            if st.button("⭐⭐⭐⭐", key=f"fb4_{i}"):
                process_feedback(question, answer, 4)
                st.success("Feedback registrado!")
        with col5:
            if st.button("⭐⭐⭐⭐⭐", key=f"fb5_{i}"):
                process_feedback(question, answer, 5)
                st.success("Feedback registrado!")

    # Campo de entrada de pergunta
    question = st.text_input("Digite sua pergunta:", key="question_input")

    # Botão de envio
    if st.button("Enviar Pergunta"):
        if question:
            try:
                with st.spinner("Processando sua pergunta..."):
                    result = st.session_state.qa_chain.invoke({
                        "question": question,
                        "chat_history": st.session_state.chat_history
                    })
                    
                    # Adiciona à história
                    st.session_state.chat_history.append((question, result['answer']))
                    
                    # Recarrega a página para mostrar a nova mensagem
                    st.rerun()
            except Exception as e:
                st.error(f"Erro ao processar a pergunta: {str(e)}")
        else:
            st.warning("Por favor, digite uma pergunta.")

    # Footer
    st.markdown("---")
    st.markdown("Desenvolvido com ❤️ usando LangChain e Streamlit")

else:
    st.warning("Aguarde a inicialização do sistema...")