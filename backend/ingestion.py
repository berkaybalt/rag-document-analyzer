import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from vectorstore import add_documents


def load_pdf_text_with_pages(file_path: str) -> list[tuple[str, int]]:
    """Read a PDF file and return list of (text, page_number) tuples."""
    reader = PdfReader(file_path)
    pages_text = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages_text.append((text, page_num))
    return pages_text


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
    pages_text = load_pdf_text_with_pages(file_path)
    
    source = os.path.basename(file_path)
    chunks = []
    metadatas = []
    chunk_counter = 0
    
    # Process each page separately to maintain accurate page tracking
    for page_text, page_num in pages_text:
        # Skip empty pages
        if not page_text.strip():
            continue
            
        page_chunks = split_text(page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        for chunk in page_chunks:
            chunks.append(chunk)
            metadatas.append({
                "i": chunk_counter, 
                "source": source, 
                "page": page_num,
                "text": chunk[:200]  # Store first 200 chars for highlighting
            })
            chunk_counter += 1
    
    add_documents(chunks, metadatas=metadatas)
    print(f"Added {len(chunks)} chunks from {source}")
    return len(chunks)
