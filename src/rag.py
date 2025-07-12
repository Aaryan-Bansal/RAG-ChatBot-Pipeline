import os
import json
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from notebooks.process_and_store import preprocess_and_store_chunks
from langchain.prompts import PromptTemplate

def get_rag_chain():
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    if os.path.exists("vectordb/index.faiss") and os.path.exists("vectordb/index.pkl"):
        vectorstore = FAISS.load_local("vectordb", embeddings, allow_dangerous_deserialization=True)
    else:
        chunks = preprocess_and_store_chunks()
        vectorstore = FAISS.from_texts(chunks, embeddings)
        os.makedirs("vectordb", exist_ok=True)
        vectorstore.save_local("vectordb")
        
    
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are an AI assistant. Answer the question using only the context below.
If the answer is not in the context, reply:
"I'm sorry, I could not find that information in the document."

Context:
{context}

Question: {question}

Answer:
"""
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = Ollama(model="llama2-uncensored")
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template}
    )
    
    return rag_chain, len(vectorstore.docstore._dict)
