import os
import uuid
from pathlib import Path
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredMarkdownLoader, UnstructuredWordDocumentLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from embeddings import get_embedding_model
from config import CHROMA_DB_PATH

embedding = get_embedding_model()
vectordb = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding, collection_name='rag_collection')
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

LOADER_MAPPING = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".doc": UnstructuredWordDocumentLoader,
}


def get_loader(file_path: str):
    ext = Path(file_path).suffix.lower()
    if ext not in LOADER_MAPPING:
        raise ValueError(f"不支持的格式: {ext}")
    return LOADER_MAPPING[ext](file_path)


def add_document(file_path: str, original_name: str = None) -> str:
    doc_id = str(uuid.uuid4())
    loader = get_loader(file_path)
    docs = loader.load()
    splits = text_splitter.split_documents(docs)
    for split in splits:
        split.metadata.update({"doc_id": doc_id, "source": original_name or Path(file_path).name})
    ids = [f"{doc_id}_{i}" for i in range(len(splits))]
    vectordb.add_documents(documents=splits, ids=ids)
    return doc_id


def update_document(doc_id: str, new_file_path: str, original_name: str = None):
    delete_document(doc_id)
    return add_document(new_file_path, original_name)


def delete_document(doc_id: str) -> bool:
    results = vectordb.get(where={"doc_id": doc_id})
    if results["ids"]:
        vectordb.delete(ids=results["ids"])
        return True
    return False


def list_documents():
    results = vectordb.get(include=["metadatas"])
    docs = {}
    for meta in results["metadatas"]:
        doc_id = meta.get("doc_id")
        if doc_id and doc_id not in docs:
            docs[doc_id] = {"doc_id": doc_id, "filename": meta["source"], "chunks": 0}
        if doc_id:
            docs[doc_id]["chunks"] += 1
    return list(docs.values())


def get_document_content(doc_id: str):
    results = vectordb.get(where={"doc_id": doc_id}, include=["documents"])
    return " ".join(results["documents"][:3]) + "..." if results["documents"] else ""
