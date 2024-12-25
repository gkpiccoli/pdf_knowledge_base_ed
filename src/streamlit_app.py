import streamlit as st
import plotly.express as px
from datetime import datetime
import PyPDF2
import io
import pandas as pd
from qa_system import QASystem
from pathlib import Path
from typing import List, Tuple

# Configurações de diretórios usando Path
DIRECTORIES = {
    'PDF_DIR': Path('pdfs'),
    'EXTRACTED_DIR': Path('extracted_texts'),
    'DATA_DIR': Path('data'),
    'LOGS_DIR': Path('logs'),
    'EXPORTS_DIR': Path('exports')
}

# Configurações iniciais do Streamlit
st.set_page_config(
    page_title="Sistema Avançado de QA",
    layout="wide",
    page_icon="📚",
    menu_items={
        'Get Help': 'https://github.com/seu-repositorio',
        'Report a bug': "https://github.com/seu-repositorio/issues",
        'About': "# Sistema Avançado de QA\nVersão 1.0"
    }
)

# Estilo CSS aprimorado
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
    .pdf-list {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #2ecc71;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
</style>
""", unsafe_allow_html=True)

def create_required_directories() -> None:
    """Cria todos os diretórios necessários"""
    for directory in DIRECTORIES.values():
        directory.mkdir(parents=True, exist_ok=True)

def get_pdf_files() -> List[Path]:
    """Retorna lista de arquivos PDF no diretório"""
    return list(DIRECTORIES['PDF_DIR'].glob('*.pdf'))

def process_uploaded_file(uploaded_file) -> Tuple[bool, str]:
    """Processa arquivo PDF enviado"""
    try:
        if uploaded_file.type == "application/pdf":
            # Salvar PDF
            pdf_path = DIRECTORIES['PDF_DIR'] / uploaded_file.name
            pdf_path.write_bytes(uploaded_file.getvalue())
            
            # Extrair texto
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.getvalue()))
            text = ""
            
            # Adicionar barra de progresso
            progress_bar = st.progress(0)
            total_pages = len(pdf_reader.pages)
            
            for idx, page in enumerate(pdf_reader.pages):
                text += page.extract_text() + "\n"
                progress_bar.progress((idx + 1) / total_pages)
            
            # Salvar texto extraído
            text_path = DIRECTORIES['EXTRACTED_DIR'] / f"{uploaded_file.name.replace('.pdf', '.txt')}"
            text_path.write_text(text, encoding="utf-8")
            
            progress_bar.empty()
            return True, f"Arquivo {uploaded_file.name} processado com sucesso!"
    except Exception as e:
        return False, f"Erro ao processar arquivo: {str(e)}"

def load_feedback_data() -> pd.DataFrame:
    """Carrega dados de feedback do sistema"""
    feedback_file = DIRECTORIES['DATA_DIR'] / 'feedback.json'
    if feedback_file.exists():
        try:
            feedback_data = pd.read_json(feedback_file)
            feedback_data['timestamp'] = pd.to_datetime(feedback_data['timestamp'])
            return feedback_data
        except Exception as e:
            st.error(f"Erro ao carregar dados de feedback: {str(e)}")
    return pd.DataFrame()

def show_metrics_dashboard():
    """Exibe o dashboard com métricas do sistema"""
    st.header("📊 Dashboard do Sistema", divider="rainbow")

    # Carregar dados
    feedback_data = load_feedback_data()
    pdf_files = get_pdf_files()

    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📚 Documentos", len(pdf_files))
    
    with col2:
        total_questions = len(feedback_data) if not feedback_data.empty else 0
        st.metric("❓ Perguntas", total_questions)
    
    with col3:
        avg_rating = feedback_data['rating'].mean() if not feedback_data.empty else 0.0
        st.metric("⭐ Média Feedback", f"{avg_rating:.1f}")
    
    with col4:
        total_pages = sum(len(PyPDF2.PdfReader(str(pdf)).pages) for pdf in pdf_files) if pdf_files else 0
        st.metric("📄 Total Páginas", total_pages)

    # Lista de documentos
    with st.expander("📁 Documentos Disponíveis", expanded=True):
        if pdf_files:
            for pdf_file in pdf_files:
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"📄 {pdf_file.name}")
                with col2:
                    st.write(f"Tamanho: {pdf_file.stat().st_size / 1024:.1f} KB")
                with col3:
                    if st.button("🗑️ Remover", key=f"remove_{pdf_file.name}"):
                        try:
                            pdf_file.unlink()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Erro ao remover arquivo: {str(e)}")
        else:
            st.info("Nenhum documento PDF disponível.")

    # Gráficos de feedback
    if not feedback_data.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            fig_feedback = px.box(
                feedback_data,
                y='rating',
                title='📊 Distribuição das Avaliações'
            )
            st.plotly_chart(fig_feedback, use_container_width=True)
        
        with col2:
            fig_timeline = px.line(
                feedback_data,
                x='timestamp',
                y='rating',
                title='📈 Histórico de Avaliações'
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

def chat_interface():
    """Interface de chat do sistema"""
    st.header("💬 Chat Interativo", divider="rainbow")
    
    # Campo de entrada
    if prompt := st.chat_input("Digite sua pergunta..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        try:
            # Adicionar spinner durante o processamento
            with st.spinner('Processando sua pergunta...'):
                response = st.session_state.qa_system.process_query(prompt)
            
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
            
            # Área de feedback
            with st.expander("📝 Feedback"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    rating = st.slider("Avaliação:", 1, 5, 3)
                    comment = st.text_area("Comentário (opcional):")
                with col2:
                    if st.button("Enviar"):
                        st.session_state.qa_system.process_feedback(
                            prompt,
                            response["answer"],
                            rating,
                            comment
                        )
                        st.success("Feedback enviado!")
            
            # Exibir fontes
            if "sources" in response and response["sources"]:
                with st.expander("📚 Fontes"):
                    for idx, source in enumerate(response["sources"], 1):
                        st.markdown(f"**Fonte {idx}:**")
                        st.markdown(f"```{source[:200]}...```")
        
        except Exception as e:
            st.error(f"Erro: {str(e)}")

    # Exibir histórico
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

def main():
    # Inicialização
    create_required_directories()
    
    if 'qa_system' not in st.session_state:
        st.session_state.qa_system = QASystem()
        st.session_state.messages = []

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        # Upload de arquivos
        st.subheader("📁 Upload de Documentos")
        uploaded_files = st.file_uploader(
            "Carregar PDFs",
            type="pdf",
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                success, message = process_uploaded_file(uploaded_file)
                if success:
                    st.success(message)
                else:
                    st.error(message)
        
        # Botões de controle
        if st.button("🔄 Reiniciar Chat"):
            st.session_state.messages = []
            st.session_state.qa_system.clear_chat_history()
            st.success("Chat reiniciado!")
        
        if st.button("💾 Exportar Histórico"):
            filename = f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = DIRECTORIES['EXPORTS_DIR'] / filename
            st.session_state.qa_system.export_chat_history(str(filepath))
            st.success(f"Histórico exportado: {filename}")
    
    # Área principal
    tab1, tab2 = st.tabs(["💬 Chat", "📊 Dashboard"])
    
    with tab1:
        chat_interface()
    
    with tab2:
        show_metrics_dashboard()

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center'>
            <h4>💡 Dicas de Uso</h4>
            <ul style='list-style-type: none'>
                <li>✨ Faça perguntas específicas para respostas mais precisas</li>
                <li>📝 Use o feedback para nos ajudar a melhorar</li>
                <li>📊 Explore o dashboard para ver estatísticas</li>
            </ul>
            <p>Desenvolvido com ❤️ usando LangChain e Streamlit</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()