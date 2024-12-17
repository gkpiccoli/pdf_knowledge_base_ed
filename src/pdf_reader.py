from PyPDF2 import PdfReader
import os
import re
from pathlib import Path

def clean_text(text):
    """
    Limpa o texto removendo espaços extras e caracteres especiais.
    """
    # Remove quebras de linha duplicadas
    text = re.sub(r'\n\s*\n', '\n\n', text)
    # Remove espaços múltiplos
    text = re.sub(r'\s+', ' ', text)
    # Remove espaços no início e fim
    text = text.strip()
    return text

def read_pdf(pdf_path):
    """
    Lê um arquivo PDF e retorna seu texto.
    """
    try:
        reader = PdfReader(pdf_path)
        text = ""
        total_pages = len(reader.pages)
        
        print(f"\nProcessando {Path(pdf_path).name} - {total_pages} páginas")
        
        for i, page in enumerate(reader.pages):
            print(f"Lendo página {i+1}/{total_pages}", end='\r')
            page_text = page.extract_text()
            text += page_text + "\n"
        
        print(f"\nConcluído: {Path(pdf_path).name}")
        return clean_text(text)
    
    except Exception as e:
        print(f"Erro ao ler {pdf_path}: {str(e)}")
        return ""

def process_pdf_directory(directory):
    """
    Processa todos os PDFs em um diretório.
    """
    if not os.path.exists(directory):
        print(f"ERRO: Diretório não encontrado: {directory}")
        return {}

    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"ERRO: Nenhum arquivo PDF encontrado em: {directory}")
        return {}

    print(f"\nEncontrados {len(pdf_files)} arquivos PDF:")
    for pdf in pdf_files:
        print(f"- {pdf}")
    print("")
    
    texts = {}
    for filename in pdf_files:
        filepath = os.path.join(directory, filename)
        try:
            texts[filename] = read_pdf(filepath)
            
            text_length = len(texts[filename])
            words = len(texts[filename].split())
            print(f"\nEstatísticas para {filename}:")
            print(f"- Caracteres: {text_length}")
            print(f"- Palavras: {words}")
            print("-" * 50)
            
        except Exception as e:
            print(f"Erro ao processar {filename}: {str(e)}")
    
    return texts

def save_texts(texts, output_dir):
    """
    Salva os textos extraídos em arquivos TXT.
    """
    if not texts:
        print("Nenhum texto para salvar.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    for filename, text in texts.items():
        txt_filename = os.path.splitext(filename)[0] + '.txt'
        txt_path = os.path.join(output_dir, txt_filename)
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Texto salvo em: {txt_path}")

def main():
    """
    Função principal que coordena o processamento dos PDFs.
    """
    # Obtém o caminho absoluto do diretório base do projeto
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Define os caminhos absolutos para os diretórios
    pdf_dir = os.path.join(base_dir, "pdfs")
    output_dir = os.path.join(base_dir, "extracted_texts")
    
    print("="*50)
    print("INICIANDO PROCESSAMENTO DE PDFs")
    print("="*50)
    print(f"Diretório base: {base_dir}")
    print(f"Diretório de PDFs: {pdf_dir}")
    print(f"Diretório de saída: {output_dir}")
    print("="*50)
    
    # Verifica se o diretório de PDFs existe
    if not os.path.exists(pdf_dir):
        print(f"ERRO: Diretório de PDFs não encontrado: {pdf_dir}")
        print("Criando diretório pdfs...")
        os.makedirs(pdf_dir)
    
    # Lista os arquivos no diretório de PDFs
    if os.path.exists(pdf_dir):
        files = os.listdir(pdf_dir)
        print("\nArquivos encontrados no diretório pdfs:")
        for f in files:
            print(f"- {f}")
    
    # Processa os PDFs
    results = process_pdf_directory(pdf_dir)
    
    # Salva os textos extraídos
    save_texts(results, output_dir)
    
    # Mostra o número de PDFs processados
    print("\n" + "="*50)
    print(f"Total de PDFs processados: {len(results)}")
    print("="*50)

if __name__ == "__main__":
    main()