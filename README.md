# PDF Knowledge Base QA System

Sistema de perguntas e respostas baseado em documentos PDF usando LangChain e Ollama.

## Pré-requisitos

- Python 3.8+
- Ollama instalado e rodando localmente
- Modelo Mistral configurado no Ollama

## Instalação

1. Clone o repositório
```bash
git clone [URL_DO_SEU_REPOSITORIO]
cd pdf_knowledge_base

sudo apt install -y build-essential git curl wget python3-pip

## Crie um ambiente virtual

python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
.\venv\Scripts\activate  # Windows

## Instale as dependências

pip install -r requirements.txt
