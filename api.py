from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import re
import sympy
import faiss
import torch
import ollama
import os
import logging
import fitz 
from fastapi.responses import PlainTextResponse
import numpy as np

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, Document
from llama_index.vector_stores.faiss import FaissVectorStore
from sympy import SympifyError

# Initialize FastAPI
app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set device for computation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


DEFAULT_MODEL = "qwen2-math:latest"


documents = []  
doc_names = []  
embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")



faiss_index = faiss.IndexFlatL2(1024)  # Assuming 384-d embeddings for MiniLM
vector_store = FaissVectorStore(faiss_index=faiss_index)



class MathQueryRequest(BaseModel):
    question: str


class MathQueryResponse(BaseModel):
    answer: str
    references: list[str]


class TextExtractionRequest(BaseModel):
    text: str


class IndexingRequest(BaseModel):
    docs: list[dict]  # Expecting a list of {"name": "doc_name", "content": "text"}



def create_index(docs):
    """Creates FAISS index from document texts."""
    global documents, doc_names

    for doc in docs:
        documents.append(Document(text=doc["content"]))
        doc_names.append(doc["name"])

    embeddings = np.array([embedding_model.get_text_embedding(doc["content"]) for doc in docs], dtype="float32")
    
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    if embeddings.shape[1] != faiss_index.d:
        raise ValueError(f"Embedding dimension mismatch: FAISS expects {faiss_index.d}, got {embeddings.shape[1]}")

    faiss_index.add(embeddings)
    logger.info(f"FAISS index updated with {len(docs)} documents.")

# def retrieve_documents(query, top_k=3, min_results=1, dynamic_threshold=0.3):
#    # """Retrieves the most relevant documents using FAISS while ensuring at least one result is returned."""

#     if faiss_index.ntotal == 0:
#         logger.warning("FAISS index is empty. No documents available for retrieval.")
#         return [], []

#     query_embedding = np.array(embedding_model.get_text_embedding(query), dtype="float32").reshape(1, -1)

#     if query_embedding.shape[1] != faiss_index.d:
#         raise ValueError(f"Embedding dimension mismatch: FAISS expects {faiss_index.d}, got {query_embedding.shape[1]}")

#     distances, indices = faiss_index.search(query_embedding, top_k)

#     # Sort by similarity (lower distance = more relevant)
#     sorted_results = sorted(zip(indices[0], distances[0]), key=lambda x: x[1])

#     valid_indices = []
#     for i, distance in sorted_results:
#         if 0 <= i < len(documents) and distance < dynamic_threshold:
#             valid_indices.append(i)

#     # Ensure at least one result is returned if nothing meets the threshold
#     if not valid_indices and sorted_results:
#         logger.warning("No results met the threshold. Returning the best available match.")
#         valid_indices.append(sorted_results[0][0])  # Return the best match

#     relevant_docs = [documents[i].text for i in valid_indices]
#     relevant_doc_names = list(set([doc_names[i] for i in valid_indices]))  # Remove duplicates

#     return relevant_docs, relevant_doc_names


# Function to retrieve relevant documents
def retrieve_documents(query, top_k=3):
    """Finds the most relevant documents for a given query."""
    query_embedding = np.array(embedding_model.get_text_embedding(query), dtype="float32").reshape(1, -1)

    if query_embedding.shape[1] != faiss_index.d:
        raise ValueError(f"Embedding dimension mismatch: FAISS expects {faiss_index.d}, got {query_embedding.shape[1]}")

    distances, indices = faiss_index.search(query_embedding, top_k)
    
    relevant_docs = [documents[i].text for i in indices[0] if i < len(documents)]
    relevant_doc_names = list(set([doc_names[i] for i in indices[0] if i < len(doc_names)]))  # Remove duplicates

    return relevant_docs, relevant_doc_names


# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text


# API Endpoint to Upload Multiple PDFs
@app.post("/api/upload-pdfs")
async def upload_pdfs(files: list[UploadFile] = File(...)):
    """Uploads multiple PDF files, extracts text, and indexes them."""
    try:
        docs = []
        for file in files:
            file_path = f"temp_{file.filename}"
            with open(file_path, "wb") as buffer:
                buffer.write(file.file.read())
            
            text = extract_text_from_pdf(file_path)
            os.remove(file_path)
            
            docs.append({"name": file.filename, "content": text})
        
        create_index(docs)
        return {"message": "PDFs indexed successfully", "document_count": len(docs)}
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Error processing PDFs: {err}")

def format_latex_response(response_generator):
    response_text = "".join(response_generator)  # Convert generator to string

    # Convert LaTeX delimiters to Streamlit-compatible formatting
    response_text = response_text.replace("\\(", "$").replace("\\)", "$")  # Inline math
    response_text = response_text.replace("\\[", "$$").replace("\\]", "$$")  # Block math

    # Handle environments like equation, align, gather, etc.
    response_text = re.sub(r'\\begin{equation}(.*?)\\end{equation}', r'$$\1$$', response_text, flags=re.DOTALL)
    response_text = re.sub(r'\\begin{align}(.*?)\\end{align}', r'$$\1$$', response_text, flags=re.DOTALL)
    response_text = re.sub(r'\\begin{alignat}(.*?)\\end{alignat}', r'$$\1$$', response_text, flags=re.DOTALL)
    response_text = re.sub(r'\\begin{gather}(.*?)\\end{gather}', r'$$\1$$', response_text, flags=re.DOTALL)
    response_text = re.sub(r'\\begin{CD}(.*?)\\end{CD}', r'$$\1$$', response_text, flags=re.DOTALL)

    return response_text
# API Endpoint for Querying Documents
@app.post("/api/math-query", response_class=PlainTextResponse)
async def math_query(request: MathQueryRequest):
    """Handles math queries with document retrieval."""
    try:
        response_text, doc_names_used = chat_with_docs(request.question)
        formatted_response = format_latex_response(response_text)
        references = {"references": list(set(doc_names_used))}  # Returns only unique document names
        return formatted_response + "\n\n" + str(references)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def filter_relevant_docs(prompt, docs, doc_names):
    """Uses LLM to filter out irrelevant documents before answering."""
    relevant_docs = []
    relevant_doc_names = []

    for doc, name in zip(docs, doc_names):
        relevance_prompt = f"Is the following document relevant to the query '{prompt}'?\n\n{doc}\n\nAnswer with 'yes' or 'no'."
        
        relevance_response = ollama.chat(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": relevance_prompt}]
        )["message"].get("content", "").strip().lower()

        if "yes" in relevance_response:
            relevant_docs.append(doc)
            relevant_doc_names.append(name)

    return relevant_docs, relevant_doc_names


# Function to interact with Ollama API
def chat_with_docs(prompt: str):
    """Generates AI response using Ollama and relevant documents."""
    try:
        retrieved_docs, retrieved_doc_names = retrieve_documents(prompt)
        relevant_docs, doc_names_used = filter_relevant_docs(prompt, retrieved_docs, retrieved_doc_names)

        # ✅ Step-by-Step Math Tutor Prompt
        step_by_step_prompt = (
            "You are a math tutor providing structured, step-by-step solutions. For the given query:\n"
            "1. Identify the problem type (e.g., derivative, integral, proof).\n"
            "2. Break it down logically with clear steps and justifications.\n"
            "3. Apply theorems/formulas where needed.\n"
            "4. Conclude with the final answer and verification if applicable.\n\n"
        )

        if relevant_docs:
            # ✅ Case 1: Use retrieved documents as context
            context = "\n".join(relevant_docs)
            full_prompt = f"{step_by_step_prompt}Use the following documents for reference:\n{context}\n\nQuestion: {prompt}"
            references_text = f"\n\nReferences: {list(set(doc_names_used))}"
        else:
            # ✅ Case 2: No relevant documents → Answer from general knowledge
            full_prompt = f"{step_by_step_prompt}Answer the following question based on your knowledge:\n\nQuestion: {prompt}"
            references_text = "\n\nReferences: None"

        response = ollama.chat(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": full_prompt}]
        )

        return response["message"].get("content", "No response from model."), doc_names_used

    except Exception as err:
        logger.error(f"Ollama chat error: {err}", exc_info=True)
        return f"Error: {str(err)}", []

