# **📘 Installation & Setup Guide for My Math Query System**  

This guide explains how to **install, set up, and configure** my **math-aware AI system** using **FastAPI, Ollama, FAISS, and LaTeX support**.  

---

## **1️⃣ Install Required Dependencies**  

Before running the system, make sure you have **Python 3.9+** installed. Then, install all required dependencies using:

```bash
pip install fastapi uvicorn \
    ollama faiss-cpu torch \
    sentence-transformers transformers \
    spacy pymupdf numpy sympy \
    streamlit llama-index llama-index-llms-ollama \
    llama-index-vector-stores-faiss faiss-cpu llama-index-embeddings-huggingface

python -m spacy download en_core_web_sm

```

---

## **2️⃣ Install & Configure Ollama**
Ollama is used for **step-by-step math explanations**. To install it:

1️⃣ Install **Ollama** (if not already installed):  
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

2️⃣ **Pull the Required Math Model:**  
I use **`qwen2-math:latest`** as the primary model:
```bash
ollama pull qwen2-math:latest
```
---

## **3️⃣ Set Up Environment Variables**
Since my system requires an **OpenAI API Key** for some functions, you should store it securely:

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

To make it permanent, add it to your **`.bashrc`** or **`.zshrc`**:
```bash
echo 'export OPENAI_API_KEY="your-openai-api-key"' >> ~/.bashrc
source ~/.bashrc  # Apply changes
```

---

## **4️⃣ Start the FastAPI Server**
Once everything is installed, navigate to the **project directory** and run:

```bash
uvicorn api:app --reload
```

If your API file is inside a folder (`src/`), modify the command:
```bash
uvicorn src.api:app --reload
```

💡 **If you get an import error (`Could not import module "api"`), ensure you're in the correct directory.**  

---

## **6️Example API Requests**
### **📌 Upload PDFs for Indexing**
To extract text from PDFs and store it for retrieval:

```bash
curl -X 'POST' 'http://127.0.0.1:8000/api/upload-pdfs' \
-H 'accept: application/json' \
-F 'files=@sample.pdf'
```

---

### **📌 Ask a Math Question**
#### **Example: Find the Derivative**
```bash
curl -X 'POST' 'http://127.0.0.1:8000/api/math-query' \
-H 'accept: text/plain' \
-H 'Content-Type: application/json' \
-d '{"question": "Find the derivative of the function $f(x) = x^3 + 3x^2 - 5x + 7$."}'
```






## **📌 Math-Focused Enhancements**  

I've built a **math-aware question-answering system** that integrates **LaTeX processing, FAISS-based retrieval, symbolic computation, and structured step-by-step solutions**. Here's how I implemented each enhancement:

---

### **1️⃣ LaTeX-Aware Retrieval**  
✅ I extended the **retrieval pipeline** to **index math expressions**, making searches for formulas and theorems more accurate.  
✅ I use **regular expressions (`re.findall(r'\$(.*?)\$')`)** to extract **LaTeX math expressions** from documents.  
✅ These expressions are then **parsed using `sympy.sympify()`**, allowing **symbolic matching** in the retrieval step.  
✅ The **FAISS index stores both text and math embeddings**, ensuring that queries involving **formulas, equations, or theorems** return **contextually relevant results**.  

**📌 Example Implementation in My Code:**
```python
def extract_math_expressions(text):
    math_expressions = re.findall(r'\$(.*?)\$', text)
    sympy_expressions = [sympy.sympify(expr) for expr in math_expressions if expr]
    return sympy_expressions
```

---

### **2️⃣ Advanced Text Processing**  
✅ I implemented **symbolic computation** using `sympy` to **parse and solve** mathematical expressions.  
✅ If a query contains **LaTeX math expressions**, my system **converts them into a structured form** for processing.  
✅ I also integrated **spaCy NLP** to **detect named theorems** and mathematical laws in documents.  

**📌 Example Implementation in My Code:**
```python
def solve_math_query(query):
    extracted_expr, _ = extract_math_expressions(query)
    if extracted_expr:
        results = [sympy.simplify(expr) for expr in extracted_expr]
        return results
    return None
```
**💡 Benefit:** This allows my system to **interpret and manipulate math expressions** rather than just performing simple text-based retrieval.

---

### **3️⃣ Improved UI (Math-Friendly Display with LaTeX)**  
✅ I ensured that **queries and responses** maintain **LaTeX formatting**, making the output **clear and readable**.  
✅ My system automatically **converts responses into properly formatted LaTeX** before displaying them.  
✅ This improves **the user experience**, especially for complex equations and derivations.

**📌 Example Implementation in My Code:**
```python
def format_latex_response(response_generator):
    response_text = "".join(response_generator)
    response_text = response_text.replace("\\(", "$").replace("\\)", "$")
    response_text = response_text.replace("\\[", "$$").replace("\\]", "$$")
    return response_text
```
**💡 Benefit:** This ensures that **math expressions are displayed properly**, making the system more useful for students and researchers.

---

### **4️⃣ Semantic Search Optimization**  
✅ I optimized **semantic search for math-related queries** by using **math-aware embeddings**.  
✅ I implemented **FAISS Approximate Nearest Neighbor (ANN) search** to improve **query relevance**.  

**📌 Example Implementation in My Code:**
```python
faiss_index = faiss.IndexFlatL2(768)  # Optimized for 768-d embeddings
vector_store = FaissVectorStore(faiss_index=faiss_index)
index = VectorStoreIndex.from_documents(formatted_documents, vector_store=vector_store)
```
**💡 Benefit:** This ensures that my system **retrieves the most relevant mathematical content**, whether it’s a theorem, proof, or equation.

---

### **5️⃣ Step-by-Step Solutions (Tutor-Like Explanations)**  
✅ I designed the system to **break down solutions step by step**, just like a math tutor would.  
✅ Instead of just giving an answer, my system **identifies the problem type, applies the correct method, and explains each step clearly**.  
✅ I implemented this using **prompt engineering with Ollama**, ensuring that responses are structured properly.

**📌 Example Implementation in My Code:**
```python
step_by_step_prompt = f"""
You are a math tutor providing structured, step-by-step solutions. For the given query:
1. Identify the problem type (e.g., derivative, integral, proof).
2. Break it down logically with clear steps and justifications.
3. Apply theorems/formulas where needed.
4. Conclude with the final answer and verification if applicable.

Query: {prompt}
"""
```
**💡 Benefit:** This makes my system **not just a search engine, but an actual math tutor**, helping users **understand concepts rather than just getting answers**.

