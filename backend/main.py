from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from pathlib import Path
from pypdf import PdfReader, PdfWriter
from io import BytesIO

from rag import query as rag_query
from ingestion import ingest_pdf
from vectorstore import list_documents, delete_document, clear_all

app = FastAPI(title="Document RAG API", version="1.1")


class ChatRequest(BaseModel):
    message: str


# Connect
llm = ChatOllama(model="llama3.2", base_url="http://localhost:11434")


@app.get("/")
def read_root():
    return {"status": "Backend working!"}


@app.get("/test-ai")
def test_ai():
    try:
        response = llm.invoke("Test.")
        return {
            "status": "Success!",
            "model": "llama3.2",
            "answer": response.content,
        }
    except Exception as e:
        return {
            "status": "Error",
            "message": "No response",
            "detail": str(e),
        }


@app.post("/chat")
def chat(payload: ChatRequest):
    try:
        result = rag_query(payload.message)
        return {
            "status": "Success!",
            "answer": result["answer"],
            "sources": result["sources"],
        }
    except Exception as e:
        return {
            "status": "Error",
            "message": "Chat failed",
            "detail": str(e),
        }


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    try:
        # Save uploaded file to a temporary path under backend/
        temp_path = file.filename
        contents = await file.read()
        with open(temp_path, "wb") as f:
            f.write(contents)

        chunks = ingest_pdf(temp_path)
        return {
            "status": "Success!",
            "filename": file.filename,
            "chunks": chunks,
        }
    except Exception as e:
        return {
            "status": "Error",
            "message": "Ingestion failed",
            "detail": str(e),
        }


@app.get("/documents")
def get_documents():
    """List all ingested document sources (filenames)."""
    try:
        docs = list_documents()
        return {"status": "Success!", "documents": docs}
    except Exception as e:
        return {
            "status": "Error",
            "message": "Could not list documents",
            "detail": str(e),
        }


@app.delete("/documents/{source}")
def delete_document_endpoint(source: str):
    """Delete all chunks belonging to the given source filename."""
    try:
        delete_document(source)
        return {"status": "Success!", "deleted": source}
    except Exception as e:
        return {
            "status": "Error",
            "message": "Could not delete document",
            "detail": str(e),
        }


@app.post("/reset")
def reset_vectorstore():
    """Reset the entire vector store. WARNING: Deletes all ingested documents!"""
    try:
        clear_all()
        return {"status": "Success!", "message": "Vector store cleared. You can now re-ingest documents."}
    except Exception as e:
        return {
            "status": "Error",
            "message": "Could not reset vector store",
            "detail": str(e),
        }


@app.get("/pdf/{filename}")
def serve_pdf(filename: str, search: str = None, page: int = None):
    """
    Serve raw PDF file with default viewer.
    Browser's native PDF viewer will handle rendering.
    """
    try:
        backend_dir = Path(__file__).resolve().parent
        pdf_path = backend_dir / filename
        
        if not pdf_path.exists():
            pdf_path = backend_dir.parent / filename
        
        if pdf_path.exists() and pdf_path.suffix.lower() == ".pdf":
            # Serve raw PDF - browser will use default viewer
            return FileResponse(
                pdf_path, 
                media_type="application/pdf",
                headers={"Content-Disposition": 'inline'}
            )
        else:
            return {"status": "Error", "message": "PDF not found"}
    except Exception as e:
        return {"status": "Error", "message": str(e)}
