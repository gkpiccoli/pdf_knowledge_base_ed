import streamlit as st
import plotly.express as px
import pandas as pd
from pathlib import Path
from qa_system import QASystem
import json

# Configuração da página
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
    .pdf-list {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
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
    .stProgress > div > div > div > div {
        background-color: #2ecc71;
    }
    .feedback-section {
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-top: 1rem;
    }
    .source-section {
        border-left: 3px solid #2ecc71;
        padding-left: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_feedback_data():
    """Carrega dados de feedback do sistema"""
    try:
        feedback_file = Path('data/feedback.json')
        if feedback_file.exists():
            with open(feedback_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Erro ao carregar dados de feedback: {str(e)}")
    return pd.DataFrame()

def show_metrics_dashboard():
    """Exibe o dashboard com métricas do sistema"""
    st.header("📊 Dashboard do Sistema", divider="rainbow")

    # Carregar dados
    feedback_data = load_feedback_data()
    pdf_files = list(Path('pdfs').glob('*.pdf'))

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
        if pdf_files:
            import PyPDF2
            total_pages = sum(len(PyPDF2.PdfReader(str(pdf)).pages) for pdf in pdf_files)
            st.metric("📄 Total Páginas", total_pages)
        else:
            st.metric("📄 Total Páginas", 0)

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
                    if st.button("🗑️", key=f"remove_{pdf_file.name}"):
                        try:
                            pdf_file.unlink()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Erro ao remover arquivo: {str(e)}")
        else:
            st.info("Nenhum documento PDF disponível na pasta 'pdfs'.")

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
            feedback_data['timestamp'] = pd.to_datetime(feedback_data['timestamp'])
            fig_timeline = px.line(
                feedback_data,
                x='timestamp',
                y='rating',
                title='📈 Histórico de Avaliações'
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

def initialize_session_state():
    """Inicializa o estado da sessão"""
    if 'qa_system' not in st.session_state:
        st.session_state.qa_system = QASystem()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'current_model' not in st.session_state:
        st.session_state.current_model = "mistral:7b-instruct"

def main():
    initialize_session_state()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        # Configurações do modelo
        st.subheader("🤖 Configurações do Modelo")
        st.slider(
            "Temperatura",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Controla a criatividade das respostas"
        )
        
        st.session_state.k_documents = st.slider(
            "Número de documentos para contexto",
            min_value=2,
            max_value=10,
            value=6,
            help="Quantidade de documentos usados para gerar resposta"
        )
        
        # Botões de controle
        st.subheader("🔄 Controles")
        if st.button("Reindexar Documentos"):
            with st.spinner("Reindexando documentos..."):
                st.session_state.qa_system.reindex_documents()
            st.success("Base de conhecimento atualizada!")
        
        if st.button("Limpar Chat"):
            st.session_state.messages = []
            st.session_state.qa_system.clear_chat_history()
            st.success("Chat reiniciado!")
        
        if st.button("Exportar Histórico"):
            filepath = st.session_state.qa_system.export_chat_history()
            st.success(f"Histórico exportado para: {filepath}")

    # Área principal
    tab1, tab2 = st.tabs(["💬 Chat", "📊 Dashboard"])
    
    with tab1:
        st.header("💬 Chat Interativo")
        
        # Exibir mensagens anteriores
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Campo de entrada
        if prompt := st.chat_input("Digite sua pergunta..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            
            try:
                with st.spinner("Processando..."):
                    response = st.session_state.qa_system.process_query(prompt)
                
                with st.chat_message("assistant"):
                    st.write(response["answer"])
                    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
                    
                    # Exibir fontes
                    if response.get("sources"):
                        with st.expander("📚 Fontes Utilizadas"):
                            for idx, source in enumerate(response["sources"], 1):
                                st.markdown(f"""
                                <div class="source-section">
                                    <strong>Fonte {idx}:</strong> {source['source']}<br>
                                    <small>{source['created_at']}</small><br>
                                    <code>{source['content'][:200]}...</code>
                                </div>
                                """, unsafe_allow_html=True)
                
                # Área de feedback
                with st.expander("📝 Fornecer Feedback"):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        rating = st.slider("Avalie a resposta:", 1, 5, 3)
                        comment = st.text_area("Comentário (opcional):")
                    with col2:
                        if st.button("Enviar Feedback"):
                            st.session_state.qa_system.process_feedback(
                                prompt,
                                response["answer"],
                                rating,
                                comment
                            )
                            st.success("Feedback enviado com sucesso!")
            
            except Exception as e:
                st.error(f"Erro ao processar pergunta: {str(e)}")
    
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