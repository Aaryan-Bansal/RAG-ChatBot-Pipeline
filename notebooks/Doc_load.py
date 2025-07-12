from langchain_community.document_loaders import PyPDFLoader
import os

def doc_loader():
    loader = PyPDFLoader("data/AI Training Document.pdf")
    documents = loader.load()
    text = "\n".join(doc.page_content for doc in documents)

    os.makedirs("data", exist_ok=True)
    with open("data/document.txt", "w", encoding="utf-8") as f:
        f.write(text)

if __name__ == "__main__":
    doc_loader()