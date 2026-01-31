from fastapi import FastAPI
from langchain_ollama import ChatOllama

app = FastAPI(title="Document RAG API", version="1.1")

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
            "answer": response.content
        }
    except Exception as e:
        return {
            "status": "Error", 
            "message": "No response",
            "detail": str(e)
        }