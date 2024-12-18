import streamlit as st
import os
from datetime import datetime
from qa_system import create_directories, setup_qa_chain, process_feedback

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

# Inicialização das variáveis de sessão
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'qa_chain' not in st.session_state:
    create_directories()
    st.session_state.qa_chain = setup_qa_chain()

# Título principal
st.title("📚 Sistema de Perguntas e Respostas")

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
        # Lógica de exportação
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = os.path.join("exports", f"chat_history_{timestamp}.json")
        # Implementar função de exportação aqui
        st.success(f"Histórico exportado para {export_path}")

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
    with col2:
        if st.button("⭐⭐", key=f"fb2_{i}"):
            process_feedback(question, answer, 2)
    with col3:
        if st.button("⭐⭐⭐", key=f"fb3_{i}"):
            process_feedback(question, answer, 3)
    with col4:
        if st.button("⭐⭐⭐⭐", key=f"fb4_{i}"):
            process_feedback(question, answer, 4)
    with col5:
        if st.button("⭐⭐⭐⭐⭐", key=f"fb5_{i}"):
            process_feedback(question, answer, 5)

# Campo de entrada de pergunta
question = st.text_input("Digite sua pergunta:", key="question_input")

# Botão de envio
if st.button("Enviar Pergunta"):
    if question:
        with st.spinner("Processando sua pergunta..."):
            # Obtém a resposta usando o qa_chain
            result = st.session_state.qa_chain.invoke({
                "question": question,
                "chat_history": st.session_state.chat_history
            })
            
            # Adiciona à história
            st.session_state.chat_history.append((question, result['answer']))
            
            # Recarrega a página para mostrar a nova mensagem
            st.experimental_rerun()
    else:
        st.warning("Por favor, digite uma pergunta.")

# Footer
st.markdown("---")
st.markdown("Desenvolvido com ❤️ usando LangChain e Streamlit")