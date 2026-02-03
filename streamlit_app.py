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
    
    # Reset button
    if st.button("Delete All Documents)", use_container_width=True, type="secondary"):
        try:
            resp = requests.post(f"{API_BASE}/reset", timeout=60)
            data = resp.json()
            if data.get("status") == "Success!":
                st.success("Vector store cleared!")
                st.rerun()
            else:
                st.error(f"Reset failed: {data.get('message')}")
        except Exception as e:
            st.error(f"Reset error: {e}")
    
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
                                st.rerun()
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
        for i, src in enumerate(msg.get("sources", []), start=1):
            source_file = (src.get("metadata") or {}).get("source", "?")
            page_num = (src.get("metadata") or {}).get("page", 0)
            content_preview = src.get("content", "")[:100]
            st.markdown(f"**Source {i}:** [{source_file} (page {page_num + 1})](http://localhost:8000/pdf/{source_file}#page={page_num}) | *{content_preview}...*")

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
                    
                    # Display sources with links and page info
                    with st.expander("Sources", expanded=True):
                        for i, src in enumerate(sources, start=1):
                            source_file = (src.get("metadata") or {}).get("source", "?")
                            page_num = (src.get("metadata") or {}).get("page", 0)
                            content_preview = src.get("content", "")[:150]
                            distance = src.get("distance", 0)
                            
                            # Calculate similarity score (1 - normalized distance)
                            similarity = max(0, 1 - distance)
                            
                            pdf_url = f"http://localhost:8000/pdf/{source_file}#page={page_num+1}"
                            
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"**Source {i}:** [{source_file}]({pdf_url}) — Page {page_num + 1}")
                            with col2:
                                st.metric("Score", f"{similarity:.3f}", delta=None)
                            
                            st.code(content_preview + "...", language="text")

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
