# NIT Durgapur CSE Syllabus RAG Assistant

**🚀 Live Demo: [Try the App Here!](https://snehasaha1101-cse-syllabus-rag-assistant-app-xfimen.streamlit.app/)**
# Context-Aware Syllabus RAG Assistant

An advanced Retrieval-Augmented Generation (RAG) system built to query a highly structured, 161-page Computer Science B.Tech Syllabus. This system uses **LangChain (LCEL)**, **Meta's Llama-3.3-70B (Groq)**, and a local **ChromaDB** vector store.

## Key Architectural Improvements

### 1. Two-Pass Subject Extraction
Rather than naively splitting the PDF into arbitrary overlapping chunks or page boundaries, the ingestion pipeline (`ingest.py`) implements a **two-pass context extraction**:
- **Pass 1:** Scans the document using Regex to identify authentic Course Code headers (e.g., `CSC 301 Discrete Mathematics`).
- **Pass 2:** Concatenates all lines belonging to a subject into a unified block. It also maps character offsets back to the exact PDF page numbers.
- **Result:** Every single chunk injected into ChromaDB has the exact Subject Code + Title prepended to its text, preventing the "drift" of semantic context often seen in RAG applications.

### 2. Dynamic Metadata Filtering
When a user asks a question (e.g., "What are the prerequisites for CSC301?"), the system uses regex to detect `CSC301` directly from the query and dynamically passes `filter={"subject_code": "CSC301"}` to ChromaDB. This vastly reduces the search space and improves Hit@5 accuracy.

### 3. Hallucination Prevention
Strict system prompting combined with mandatory source citations (displaying exact subject codes and page mappings in the Streamlit UI) ensures that the LLM only answers from the syllabus and explicitly refuses adversarial queries.

## Evaluation Rigor
We use a standalone evaluation test suite (`eval.py`) that tests 20 diverse QA pairs ranging from direct subject lookups to adversarial refusal queries. 
The Answer Match rate is evaluated using LLM-as-a-judge (powered by `llama-3.3-70b-versatile` to ensure independent grading against the actor model).

To account for stochastic LLM judge variance, the metrics below reflect the median/range across 3 separate evaluation runs:

<!-- METRICS:START -->
- **Hit@5 (Single Subject)**: 8/9 (88.9%)
- **Answer Match Rate**: 14/20 (70.0%)
- **Adversarial Accuracy**: 7/7 (100.0%)
<!-- METRICS:END -->

To run the evaluation yourself:
```bash
# Set your API Key
$env:GROQ_API_KEY="your-groq-api-key"

# Run the test suite
python eval.py
```
This generates `eval_results.md`, measuring Hit@5, Answer Match Rate (using LLM-as-a-judge), and Refusal Precision.

## Setup & Running

1. Install pinned dependencies:
```bash
pip install -r requirements.txt
```

2. Build the Vector Store (takes ~30-60 seconds):
```bash
python ingest.py
```

3. Launch the App:
```bash
streamlit run app.py
```
