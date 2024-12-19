import streamlit as st
import os
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import PyPDF2
import io
import pandas as pd
from qa_system import create_directories, setup_qa_chain, process_feedback, export_chat_history

# Configurações iniciais
os.environ['POSTHOG_API_KEY'] = ''
st.set_page_config(page_title="Sistema Avançado de QA", layout="wide", page_icon="📚")

# Estilo CSS personalizado
st.markdown("""
<style>
    .main {padding: 2rem;}
    .stTextInput > div > div > input {padding: 0.5rem;}
    .feedback-button {margin: 0.2rem;}
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def process_uploaded_file(uploaded_file):
    try:
        if not os.path.exists("documents"):
            os.makedirs("documents")
        
        if uploaded_file.type == "application/pdf":
            file_path = os.path.join("documents", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Extrair texto para indexação
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.getvalue()))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            # Salvar texto extraído
            text_path = os.path.join("extracted_texts", 
                                   uploaded_file.name.replace('.pdf', '.txt'))
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(text)
            
            return True, f"Arquivo {uploaded_file.name} processado com sucesso!"
    except Exception as e:
        return False, f"Erro ao processar arquivo: {str(e)}"

def initialize_system():
    try:
        with st.spinner('Inicializando o sistema...'):
            create_directories()
            qa_chain = setup_qa_chain()
            return qa_chain
    except Exception as e:
        st.error(f"Erro na inicialização: {str(e)}")
        return None

def show_metrics_dashboard():
    col1, col2, col3 = st.columns(3)
    
    # Métricas básicas
    with col1:
        st.metric("📚 Total de Documentos", 
                 len([f for f in os.listdir("documents") if f.endswith('.pdf')]))
    with col2:
        st.metric("❓ Total de Perguntas", 
                 len(st.session_state.get('chat_history', [])))
    with col3:
        st.metric("⭐ Média de Feedback", 
                 f"{st.session_state.get('feedback_avg', 0):.1f}")
    
    # Gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de perguntas por dia
        if st.session_state.get('chat_history'):
            dates = [datetime.now().strftime("%Y-%m-%d") 
                    for _ in st.session_state.chat_history]
            df_questions = pd.DataFrame({'data': dates})
            fig_questions = px.histogram(df_questions, x='data', 
                                       title="Perguntas por Dia")
            st.plotly_chart(fig_questions, use_container_width=True)
    
    with col2:
        # Gráfico de feedback
        if hasattr(st.session_state, 'feedback_history'):
            feedback_data = pd.DataFrame(st.session_state.feedback_history)
            fig_feedback = px.box(feedback_data, y='rating', 
                                title="Distribuição do Feedback")
            st.plotly_chart(fig_feedback, use_container_width=True)

def main():
    # Inicialização
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False
        st.session_state.chat_history = []
        st.session_state.feedback_history = []
        st.session_state.feedback_avg = 0
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Upload de Documentos")
        uploaded_files = st.file_uploader("Carregar PDFs", 
                                        type="pdf", 
                                        accept_multiple_files=True)
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                success, message = process_uploaded_file(uploaded_file)
                if success:
                    st.success(message)
                else:
                    st.error(message)
        
        st.header("⚙️ Configurações")
        model = st.selectbox("Modelo LLM", 
                           ["mistral", "llama2", "codellama", "phi"])
        temperature = st.slider("Temperatura", 0.0, 1.0, 0.3)
        
        if st.button("Reinicializar Sistema"):
            st.session_state.system_initialized = False
            st.rerun()
    
    # Área principal
    st.title("📚 Sistema Avançado de QA")
    
    # Inicialização do sistema
    if not st.session_state.system_initialized:
        qa_chain = initialize_system()
        if qa_chain:
            st.session_state.qa_chain = qa_chain
            st.session_state.system_initialized = True
            st.rerun()
    
    # Dashboard
    with st.expander("📊 Dashboard", expanded=True):
        show_metrics_dashboard()
    
    # Área de chat
    st.header("💬 Chat")
    
    # Histórico de mensagens
    for i, (question, answer) in enumerate(st.session_state.get('chat_history', [])):
        with st.container():
            st.info(f"Pergunta: {question}")
            with st.expander("Ver resposta e fontes", expanded=True):
                st.success("Resposta:")
                st.write(answer)
                
                # Sistema de feedback
                st.write("---")
                st.write("📝 Avaliação")
                
                feedback_col1, feedback_col2 = st.columns(2)
                with feedback_col1:
                    quality = st.select_slider(
                        "Qualidade da resposta",
                        options=["Ruim", "Regular", "Bom", "Muito Bom", "Excelente"],
                        key=f"quality_{i}"
                    )
                
                with feedback_col2:
                    feedback_text = st.text_area(
                        "Comentários (opcional)",
                        key=f"feedback_{i}",
                        height=100
                    )
                
                if st.button("Enviar Feedback", key=f"send_{i}"):
                    feedback_data = {
                        'question': question,
                        'answer': answer,
                        'quality': quality,
                        'comment': feedback_text,
                        'timestamp': datetime.now().isoformat()
                    }
                    process_feedback(feedback_data)
                    st.success("Feedback registrado! Obrigado!")
    
    # Campo de pergunta
    question = st.text_input("Digite sua pergunta:")
    
    if st.button("Enviar"):
        if question:
            with st.spinner("Processando..."):
                try:
                    result = st.session_state.qa_chain.invoke({
                        "question": question,
                        "chat_history": st.session_state.get('chat_history', [])
                    })
                    
                    st.session_state.chat_history.append((question, result['answer']))
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro: {str(e)}")
        else:
            st.warning("Por favor, digite uma pergunta.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        💡 **Dicas:**
        - Faça perguntas específicas para respostas mais precisas
        - Use o feedback para nos ajudar a melhorar
        - Explore o dashboard para ver estatísticas
        
        Desenvolvido com ❤️ usando LangChain e Streamlit
    """)

if __name__ == "__main__":
    main()