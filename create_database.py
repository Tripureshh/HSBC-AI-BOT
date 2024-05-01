
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.vectorstores import Chroma
import os
import shutil
from langchain_cohere import CohereEmbeddings
from langchain_text_splitters import CharacterTextSplitter

CHROMA_PATH = "chroma"
DATA_PATH = "data"


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    print(chunks)
    save_to_chroma(chunks)


def load_documents():
    pdfreader = []
    pdfreader.extend(PyPDFLoader('data/HDFC FAQ.pdf').load())
    #raw_text = ''
    #for i, page in enumerate(pdfreader.pages):
     #   content = page.extract_text()
     #   if content:
      #      raw_text += content
    print("PDF loaded")
    return pdfreader


def split_text(documents: list[Document]):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 800,
        chunk_overlap  = 200,
        length_function = len,
    )
    chunks = text_splitter.split_documents(documents)
    
    
    return chunks


def save_to_chroma(chunks: list[str]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    embedding_function = CohereEmbeddings()

    #embeddings = embedding_function.embed_with_retries(chunks)
    print(chunks)
    db = Chroma.from_documents(chunks, embedding_function, persist_directory=CHROMA_PATH)
    #db = Chroma.from_texts(chunks, embedding_function, persist_directory=CHROMA_PATH)
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
    os.environ["COHERE_API_KEY"] = "RZFvd31jwHnH0RVw317LnrNumRtkVZBGV9D5W0bp"
    main()