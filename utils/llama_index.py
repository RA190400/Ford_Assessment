import os
import re
import sympy
import torch
import spacy
import streamlit as st
import faiss
import utils.logs as logs

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, Document
from sympy import SympifyError
from llama_index.vector_stores.faiss import FaissVectorStore

# Load NLP model for theorem detection
nlp = spacy.load("en_core_web_sm")

# Set device globally
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

###################################
# Setup Embedding Model (Math-Aware)
###################################
@st.cache_resource(show_spinner=False)
def setup_embedding_model(model: str):
    """
    Sets up an embedding model using the Hugging Face library.

    Args:
        model (str): The name of the embedding model to use.

    Returns:
        An instance of the HuggingFaceEmbedding class, configured with the specified model and device.
    """
    logs.log.info(f"Using {DEVICE} to generate embeddings")
    try:
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=model,
            device=DEVICE,
        )
        logs.log.info(f"Embedding model created successfully")
    except Exception as err:
        logs.log.error(f"[setup_embedding_model] Failed to setup the embedding model: {err}")


###################################
# Extract Math Expressions + Theorems
###################################
def extract_math_expressions(text):
    """
    Extracts LaTeX-style math expressions and detects named theorems.
    """
    try:
        math_expressions = re.findall(r'\$(.*?)\$', text)
        sympy_expressions = []
        for expr in math_expressions:
            try:
                sympy_expressions.append(sympy.sympify(expr))
            except SympifyError:
                logs.log.warning(f"Skipping invalid math expression: {expr}")
        
        # Theorem detection
        doc = nlp(text)
        named_theorems = [ent.text for ent in doc.ents if ent.label_ in ["LAW", "EVENT"]]
        
        return sympy_expressions, named_theorems
    except Exception as err:
        logs.log.error(f"[extract_math_expressions] Error: {err}")
        return [], []

###################################
# Load and Process Documents
###################################
def load_documents(data_dir):
    """
    Loads documents and extracts math expressions and theorems.
    """
    try:
        files = SimpleDirectoryReader(input_dir=data_dir, recursive=True)
        documents = files.load_data()
        
        indexed_data = []
        for doc in documents:
            text = doc.text
            math_expressions, theorems = extract_math_expressions(text)
            indexed_data.append({"text": text, "math": math_expressions, "theorems": theorems})  
        
        logs.log.info(f"Loaded {len(documents):,} documents with math expressions & theorems.")
        return indexed_data
    except Exception as err:
        logs.log.error(f"[load_documents] Error: {err}")
        raise Exception(f"Error loading documents: {err}")

###################################
# Create Document Index
###################################
@st.cache_resource(show_spinner=False)
def create_index(documents):
    """
    Creates a FAISS-based index with mathematical expressions.
    """
    try:
        formatted_documents = [Document(text=entry["text"]) for entry in documents]
        for entry in documents:
            for formula in entry["math"]:
                formatted_documents.append(Document(text=str(formula)))
            for theorem in entry["theorems"]:
                formatted_documents.append(Document(text=theorem))
        
        faiss_index = faiss.IndexFlatL2(768)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        index = VectorStoreIndex.from_documents(formatted_documents, vector_store=vector_store)
        
        logs.log.info("[create_index] Index created successfully with math+theorem search.")
        return index
    except Exception as err:
        logs.log.error(f"[create_index] Failed: {err}")
        raise Exception(f"Index creation failed: {err}")

###################################
# Query Engine: Hybrid Search (FAISS + Symbolic Matching)
###################################
def create_query_engine(documents):
    """
    Creates a query engine that handles both text and symbolic math queries.
    """
    try:
        index = create_index(documents)
        
        if "top_k" not in st.session_state:
            st.session_state["top_k"] = 5
        if "chat_mode" not in st.session_state:
            st.session_state["chat_mode"] = "default"
        
        query_engine = index.as_query_engine(
            similarity_top_k=st.session_state["top_k"],
            response_mode=st.session_state["chat_mode"],
            streaming=True,
        )
        
        st.session_state["query_engine"] = query_engine
        logs.log.info("[create_query_engine] Query Engine initialized.")
        return query_engine
    except Exception as e:
        logs.log.error(f"[create_query_engine] Error: {e}")
        raise Exception(f"Error creating Query Engine: {e}")

###################################
# Symbolic Math Processing for Queries
###################################
def solve_math_query(query):
    """
    Checks if a query contains math and attempts to solve it symbolically.
    """
    try:
        extracted_expr, _ = extract_math_expressions(query)
        if extracted_expr:
            results = [sympy.simplify(expr) for expr in extracted_expr]
            return results
        return None
    except Exception as err:
        logs.log.error(f"[solve_math_query] Failed: {err}")
        return None

###################################
# Hybrid Query Processing
###################################
def process_query(user_query):
    """
    Processes a query using FAISS + Symbolic Math Reasoning.
    """
    symbolic_results = solve_math_query(user_query)
    text_results = st.session_state["query_engine"].query(user_query)
    
    if symbolic_results:
        return f"Symbolic Result: {symbolic_results}\n\nText-Based Search: {text_results}"
    return text_results
