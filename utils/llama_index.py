import os
import re
import sympy
import torch
import streamlit as st
import utils.logs as logs

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, Document
from sympy import SympifyError
from llama_index.vector_stores.faiss import FaissVectorStore

# Set device globally
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

###################################
#
# Setup Embedding Model
#
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
#
# Extract Math Expressions
#
###################################

def extract_math_expressions(text):
    """
    Extracts LaTeX-style math expressions from a given text.

    Args:
        text (str): The document text.

    Returns:
        list: A list of extracted math expressions.
    """
    try:
        math_expressions = re.findall(r'\$(.*?)\$', text)  # Extracts LaTeX expressions
        sympy_expressions = []
        for expr in math_expressions:
            try:
                sympy_expressions.append(sympy.sympify(expr))
            except SympifyError as err:
                logs.log.warning(f"[extract_math_expressions] Sympy Error for expression '{expr}': {err}")
        return sympy_expressions
    except Exception as err:
        logs.log.error(f"[extract_math_expressions] Error extracting math expressions: {err}")
        return []

###################################
#
# Load Documents with Math Extraction
#
###################################

def load_documents(data_dir: str):
    """
    Loads documents from a directory and extracts LaTeX math expressions.

    Args:
        data_dir (str): Directory containing the documents.

    Returns:
        A list of dictionaries, each containing 'text' (document content) and 'math' (extracted formulas).
    """
    try:
        files = SimpleDirectoryReader(input_dir=data_dir, recursive=True)
        documents = files.load_data()

        indexed_data = []
        for doc in documents:
            text = doc.text
            math_expressions = extract_math_expressions(text)
            indexed_data.append({"text": text, "math": math_expressions})  

        logs.log.info(f"[load_documents] Loaded {len(documents):,} documents with extracted math formulas.")
        return indexed_data
    except Exception as err:
        logs.log.error(f"[load_documents] Error loading documents: {err}")
        raise Exception(f"Error loading documents: {err}")

###################################
#
# Create Document Index (with Math Expressions)
#
###################################

from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

@st.cache_resource(show_spinner=False)
def create_index(documents):
    """
    Creates an index from the provided documents using FAISS for efficient nearest-neighbor search.

    Args:
        documents (list): A list of dictionaries containing 'text' and 'math' extracted from files.

    Returns:
        An instance of VectorStoreIndex, containing the indexed data.
    """
    try:
        formatted_documents = [Document(text=entry["text"]) for entry in documents]
        for entry in documents:
            for formula in entry["math"]:
                formatted_documents.append(Document(text=str(formula)))

        # ✅ Correct way to initialize FaissVectorStore
        faiss_index = faiss.IndexFlatL2(768)  # Assuming 768-d embeddings (BERT-like model)
        vector_store = FaissVectorStore(faiss_index=faiss_index)

        index = VectorStoreIndex.from_documents(
            documents=formatted_documents,
            vector_store=vector_store
        )

        logs.log.info("[create_index] Index created successfully with LaTeX-aware retrieval.")
        return index
    except Exception as err:
        logs.log.error(f"[create_index] Index creation failed: {err}")
        raise Exception(f"Index creation failed: {err}")

###################################
#
# Create Query Engine (for Text + Math)
#
###################################

def create_query_engine(documents):
    """
    Creates a query engine that forces step-by-step math solutions.

    Args:
        documents (list): A list of dictionaries containing text and extracted math expressions.

    Returns:
        An instance of QueryEngine, enforcing structured math solutions.
    """
    try:
        index = create_index(documents)

        # Ensure session state keys exist before using them
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
        logs.log.info("[create_query_engine] Query Engine created successfully with step-by-step reasoning.")
        return query_engine
    except Exception as e:
        logs.log.error(f"[create_query_engine] Error creating Query Engine: {e}")
        raise Exception(f"Error creating Query Engine: {e}")

