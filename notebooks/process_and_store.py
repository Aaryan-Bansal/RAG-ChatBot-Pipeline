def preprocess_and_store_chunks():
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    import os
    import json
    from notebooks.Doc_load import doc_loader

    if not os.path.exists("data/document.txt"):
        doc_loader()

    with open("data/document.txt", "r", encoding="utf-8") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(text)

    os.makedirs("chunks", exist_ok=True)
    with open("chunks/chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    return chunks