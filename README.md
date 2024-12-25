


# PDF Knowledge Base QA System

Sistema de perguntas e respostas baseado em documentos PDF com processamento de linguagem natural usando LangChain e Ollama.

## 📋 Pré-requisitos

### Instalação do Ollama
1. Instale o Ollama seguindo as instruções em: [Ollama Installation](https://ollama.ai/download)
2. Verifique se o Ollama está rodando:
```bash
ollama serve
```

### Instalação dos Modelos LLM
Execute os seguintes comandos para baixar os modelos necessários:

```bash
# Modelo principal para QA
ollama pull mistral:7b-instruct

# Modelo para embeddings
ollama pull nomic-embed-text
```

### Dependências Python
Instale as dependências do projeto:

```bash
pip install -r requirements.txt
```

## 🚀 Configuração do Ambiente

1. Clone o repositório:
```bash
git clone [URL_DO_SEU_REPOSITORIO]
cd pdf_knowledge_base
```

2. Crie um ambiente virtual Python:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
.\venv\Scripts\activate  # Windows
```

3. Crie a estrutura de diretórios necessária:
```
pdf_knowledge_base/
├── pdfs/            # Coloque seus PDFs aqui
├── data/           # Dados processados
├── exports/        # Exportações do sistema
├── extracted_texts/# Textos extraídos dos PDFs
└── logs/           # Logs do sistema
```

## 💻 Uso

1. Coloque seus arquivos PDF na pasta `PDF/`

2. Execute o processamento dos PDFs:
```bash
python src/pdf_processor.py
```

3. Inicie a interface Streamlit:
```bash
streamlit run src/streamlit_app.py
```

## 🛠️ Funcionalidades

- Processamento de múltiplos arquivos PDF
- Extração de texto com manutenção de formatação
- Sistema de QA baseado em contexto
- Interface web interativa
- Sistema de feedback
- Exportação de histórico de conversas
- Logging detalhado

## 📊 Estrutura do Sistema

```
src/
├── pdf_processor.py     # Processamento de PDFs
├── qa_system.py         # Sistema de QA
└── streamlit_app.py     # Interface web
```

## ⚙️ Configurações Avançadas

### Ajuste de Parâmetros

Os principais parâmetros configuráveis estão em `qa_system.py`:

- `chunk_size`: Tamanho dos chunks de texto (default: 500)
- `chunk_overlap`: Sobreposição entre chunks (default: 100)
- `temperature`: Temperatura do modelo LLM (default: 0.3)
- `k`: Número de documentos relevantes a recuperar (default: 4)

### Modelos Alternativos

Você pode usar outros modelos do Ollama:
```bash
# Listar modelos disponíveis
ollama list

# Instalar modelos alternativos
ollama pull llama2
ollama pull codellama
```

## 📝 Logging

O sistema mantém logs detalhados em `logs/`:
- Inicialização do sistema
- Processamento de documentos
- Queries e respostas
- Erros e exceções

## 🤝 Contribuindo

1. Fork o projeto
2. Crie sua Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a Branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença [SUA_LICENÇA] - veja o arquivo LICENSE.md para detalhes

## 🎯 Status do Projeto

- [x] Processamento básico de PDF
- [x] Sistema de QA
- [x] Interface Streamlit
- [ ] Suporte a múltiplos idiomas
- [ ] Melhorias na interface
- [ ] Otimização de performance

## 🚨 Troubleshooting

### Problemas Comuns

1. **Erro de conexão com Ollama:**
   ```bash
   # Verifique se o serviço está rodando
   ollama serve
   ```

2. **Memória insuficiente:**
   - Reduza o `chunk_size`
   - Processe menos documentos por vez

3. **Erros de GPU:**
   - Verifique os requisitos de CUDA
   - Use a versão CPU se necessário

## 📞 Suporte

- Abra uma issue para reportar bugs
- Discussões para features


## 🙏 Agradecimentos

- [LangChain](https://github.com/hwchase17/langchain)
- [Ollama](https://ollama.ai)
- [Streamlit](https://streamlit.io)
```



