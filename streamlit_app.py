import requests
import streamlit as st


API_BASE = "http://localhost:8000"


st.set_page_config(page_title="Clinical Protocol RAG", layout="wide")
st.title("Clinical Protocol RAG")
st.caption("Upload clinical protocols and ask questions locally with RAG.")


with st.sidebar:
    st.header("Ingest PDF")
    uploaded_files = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)

    if st.button("Ingest PDFs", use_container_width=True) and uploaded_files:
        total_chunks = 0
        ingested_files = []
        for uploaded_file in uploaded_files:
            try:
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "application/pdf",
                    )
                }
                resp = requests.post(f"{API_BASE}/ingest", files=files, timeout=300)
                data = resp.json()
                if data.get("status") == "Success!":
                    filename = data.get('filename')
                    chunks = data.get('chunks')
                    total_chunks += chunks
                    ingested_files.append(f"{filename} ({chunks} chunks)")
                else:
                    st.error(f"Ingestion failed for {uploaded_file.name}: {data.get('message')}")
                    if detail := data.get("detail"):
                        st.caption(detail)
            except Exception as e:
                st.error(f"Request error for {uploaded_file.name}: {e}")
        
        if ingested_files:
            st.success(f"Successfully ingested {len(ingested_files)} file(s) with {total_chunks} total chunks:\n" + "\n".join(ingested_files))

    st.markdown("---")
    st.header("Documents")
    try:
        resp = requests.get(f"{API_BASE}/documents", timeout=60)
        data = resp.json()
        if data.get("status") == "Success!":
            docs = data.get("documents") or []
            if not docs:
                st.write("No documents ingested yet.")
            else:
                for doc in docs:
                    cols = st.columns([3, 1])
                    cols[0].write(doc)
                    if cols[1].button("Delete", key=f"del-{doc}"):
                        try:
                            d_resp = requests.delete(
                                f"{API_BASE}/documents/{doc}", timeout=60
                            )
                            d_data = d_resp.json()
                            if d_data.get("status") == "Success!":
                                st.success(f"Deleted {doc}")
                            else:
                                st.error(f"Delete failed: {d_data.get('message')}")
                        except Exception as e:
                            st.error(f"Delete error: {e}")
        else:
            st.error(f"Could not load documents: {data.get('message')}")
    except Exception as e:
        st.error(f"Error loading documents: {e}")


st.header("Chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Render history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        for src in msg.get("sources", []):
            source_file = (src.get("metadata") or {}).get("source", "?")
            st.markdown(f"*Source: **{source_file}***")

# Chat input
if prompt := st.chat_input("Ask a question about the ingested protocol"):
    # Add user message to history
    st.session_state["messages"].append(
        {"role": "user", "content": prompt, "sources": []}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/chat", json={"message": prompt}, timeout=300
                )
                data = resp.json()
                if data.get("status") == "Success!":
                    answer = data.get("answer", "")
                    sources = data.get("sources") or []
                    st.markdown(answer)
                    for i, src in enumerate(sources, start=1):
                        source_file = (src.get("metadata") or {}).get("source", "?")
                        st.markdown(f"*Source {i}: **{source_file}***")

                    # Save assistant message with sources
                    st.session_state["messages"].append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                    )
                else:
                    msg = data.get("message", "Chat failed")
                    detail = data.get("detail")
                    st.error(msg)
                    if detail:
                        st.caption(detail)
            except Exception as e:
                st.error(f"Request error: {e}")
