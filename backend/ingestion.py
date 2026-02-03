import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from vectorstore import add_documents


def load_pdf_text(file_path: str) -> str:
    """Read a PDF file and return its full text."""
    reader = PdfReader(file_path)
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)


def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """Split text into chunks for embedding. Returns list of chunk strings."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_text(text)


def ingest_pdf(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> int:
    """Load PDF, split into chunks, add to vector store. Returns number of chunks added."""
    text = load_pdf_text(file_path)
    chunks = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    source = os.path.basename(file_path)
    metadatas = [{"i": i, "source": source} for i in range(len(chunks))]
    add_documents(chunks, metadatas=metadatas)
    print(f"Added {len(chunks)} chunks from {source}")
    return len(chunks)
