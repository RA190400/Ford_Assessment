import os
import re
import streamlit as st
import sympy
import utils.logs as logs
from sympy import SympifyError
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# This is not used but required by llama-index and must be set FIRST
os.environ["OPENAI_API_KEY"] = "sk-abc123"

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
)



###################################
#
# Setup Embedding Model
#
###################################


@st.cache_resource(show_spinner=False)
def setup_embedding_model(
    model: str,
):
    """
    Sets up an embedding model using the Hugging Face library.

    Args:
        model (str): The name of the embedding model to use.

    Returns:
        An instance of the HuggingFaceEmbedding class, configured with the specified model and device.

    Raises:
        ValueError: If the specified model is not a valid embedding model.

    Notes:
        The `device` parameter can be set to 'cpu' or 'cuda' to specify the device to use for the embedding computations. If 'cuda' is used and CUDA is available, the embedding model will be run on the GPU. Otherwise, it will be run on the CPU.
    """
    try:
        from torch import cuda
        device = "cpu" if not cuda.is_available() else "cuda"
    except:
        device = "cpu"
    finally:
        logs.log.info(f"Using {device} to generate embeddings")

    try:
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=model,
            device=device,
        )

        logs.log.info(f"Embedding model created successfully")
        
        return
    except Exception as err:
        print(f"Failed to setup the embedding model: {err}")


###################################
#
# Load Documents
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
# Create Document Index
#
###################################


@st.cache_resource(show_spinner=False)
def create_index(_documents):
    """
    Creates an index from the provided documents and service context.

    Args:
        documents (list[str]): A list of strings representing the content of the documents to be indexed.

    Returns:
        An instance of `VectorStoreIndex`, containing the indexed data.

    Raises:
        Exception: If there is an error creating the index.

    Notes:
        The `documents` parameter should be a list of strings representing the content of the documents to be indexed.
    """

    try:
        index = VectorStoreIndex.from_documents(
            documents=_documents, show_progress=True
        )

        logs.log.info("Index created from loaded documents successfully")

        return index
    except Exception as err:
        logs.log.error(f"Index creation failed: {err}")
        raise Exception(f"Index creation failed: {err}")


###################################
#
# Create Query Engine
#
###################################


# @st.cache_resource(show_spinner=False)
def create_query_engine(_documents):
    """
    Creates a query engine from the provided documents and service context.

    Args:
        documents (list[str]): A list of strings representing the content of the documents to be indexed.

    Returns:
        An instance of `QueryEngine`, containing the indexed data and allowing for querying of the data using a variety of parameters.

    Raises:
        Exception: If there is an error creating the query engine.

    Notes:
        The `documents` parameter should be a list of strings representing the content of the documents to be indexed.

        This function uses the `create_index` function to create an index from the provided documents and service context, and then creates a query engine from the resulting index. The `query_engine` parameter is used to specify the parameters of the query engine, including the number of top-ranked items to return (`similarity_top_k`) and the response mode (`response_mode`).
    """
    try:
        index = create_index(_documents)

        query_engine = index.as_query_engine(
            similarity_top_k=st.session_state["top_k"],
            response_mode=st.session_state["chat_mode"],
            streaming=True,
        )

        st.session_state["query_engine"] = query_engine

        logs.log.info("Query Engine created successfully")

        return query_engine
    except Exception as e:
        logs.log.error(f"Error when creating Query Engine: {e}")
        raise Exception(f"Error when creating Query Engine: {e}")
