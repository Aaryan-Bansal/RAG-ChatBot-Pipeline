import os
import streamlit as st
from src.rag import get_rag_chain

st.set_page_config(page_title="Amlgo RAG Chatbot", layout="wide")
st.title("Amlgo RAG Chatbot with Llama2 and HuggingFace")


rag_chain, chunk_count = get_rag_chain()


with st.sidebar:
    st.markdown("### Model Info")
    st.markdown("- **Model:** llama2-uncensored")
    st.markdown(f"- **Chunks Indexed:** {chunk_count}")
    if st.button("Reset Chat"):
        st.session_state.clear()
        st.rerun()


if "messages" not in st.session_state:
    st.session_state["messages"] = []

query = st.chat_input("Ask a question about the document...")
if query:
    with st.spinner("Thinking..."):
        result = rag_chain(query)
        st.session_state["messages"].append({"query": query, "result": result})


for msg in st.session_state["messages"]:
    st.markdown(f"**You:** {msg['query']}")
    st.markdown(f"**Bot:** {msg['result']['result']}")
    with st.expander("Sources"):
        for i, doc in enumerate(msg['result']['source_documents']):
            st.markdown(f"**Chunk {i+1}:** {doc.page_content}")