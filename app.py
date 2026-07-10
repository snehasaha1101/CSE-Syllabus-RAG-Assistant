import os
import re
import json
import streamlit as st
import numpy as np
if not hasattr(np, "long"):
    np.long = int
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

CHROMA_DB_DIR = "./chroma_db_v4"

@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR, 
        embedding_function=embeddings
    )
    return vectorstore

@st.cache_resource
def load_subjects():
    try:
        with open("subjects.json", "r") as f:
            return set(json.load(f))
    except Exception:
        return set()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_qa_chain(vectorstore, groq_api_key, subject_code=None):
    # LLM and chain are fast to instantiate; we don't cache this function
    # because subject_code changes per query and groq_api_key shouldn't be hashed with Streamlit.
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile"
    )
    
    search_kwargs = {"k": 5}
    if subject_code:
        search_kwargs["filter"] = {"subject_code": subject_code}
        
    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    system_prompt = (
        "You are an assistant for a Computer Science syllabus at NIT Durgapur. "
        "Use the following pieces of retrieved syllabus context to answer the student's question. "
        "If you don't know the answer or the context doesn't contain the answer, explicitly state: "
        "'I am sorry, but I cannot find the answer to this in the syllabus.' "
        "Do not hallucinate subjects or credits that are not in the context.\n"
        "Important Table Legend: In syllabus tables, the columns 'L T S C H' stand for: "
        "L = Lecture hours, T = Tutorial hours, S = Sessional/Practical hours, C = Credits, H = Total Contact Hours.\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "input": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)
    
    return rag_chain_with_source

st.set_page_config(page_title="CSE Syllabus RAG Assistant", page_icon="📚", layout="centered")

st.title("📚 CSE Syllabus RAG Assistant")
st.markdown("Ask questions about the NIT Durgapur CSE B.Tech Syllabus (credits, books, depth electives, course content).")

st.sidebar.header("Configuration")
groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password")
st.sidebar.markdown(
    "Get your free API key at [Groq Console](https://console.groq.com/keys)."
)

if not groq_api_key:
    st.info("Please enter your Groq API Key in the sidebar to continue.")
elif not groq_api_key.isascii():
    st.error("Your API key contains invalid hidden characters (like em-dashes). Please re-copy it cleanly from the Groq console!")
else:
    groq_api_key = groq_api_key.strip()
    with st.spinner("Loading Syllabus Vector Store..."):
        try:
            vectorstore = load_vectorstore()
            known_subjects = load_subjects()
            st.success("Vector Store Loaded Successfully!")
        except Exception as e:
            st.error(f"Error loading vector store. Have you run `python ingest.py` yet? Details: {e}")
            st.stop()

    query = st.text_input("Ask a question:")
    st.markdown("""
    **Example questions:**
    - Provide me credits of each semester for the program
    - List down the books of discrete mathematics
    - List all 3rd semester core courses.
    """)

    if st.button("Search") and query:
        with st.spinner("Retrieving and Generating..."):
            filter_code = None
            
            # Check for semester query first
            semester_match = re.search(r'\b(?:(\d)(?:st|nd|rd|th)?\s*semester|semester\s*-?\s*(I{1,3}|IV|V|VI{1,3}|VIII?|IX|X|\d))\b', query, re.IGNORECASE)
            
            if semester_match:
                # Need to normalize roman numerals to digits if they appear
                raw_sem = semester_match.group(1) or semester_match.group(2)
                roman_to_digit = {"I": "1", "II": "2", "III": "3", "IV": "4", "V": "5", "VI": "6", "VII": "7", "VIII": "8"}
                sem_num = roman_to_digit.get(raw_sem.upper(), raw_sem)
                candidate = f"SEM{sem_num}"
                if candidate in known_subjects:
                    filter_code = candidate
                    st.info(f"Detected query for Semester {sem_num}, applying metadata filter to search...")
            
            # If not a semester query, check for a course code
            if not filter_code:
                subject_match = re.search(r'\b([a-zA-Z]{2,4})\s*(\d{2,3})\b', query)
                if subject_match:
                    candidate = (subject_match.group(1) + subject_match.group(2)).upper()
                    if candidate in known_subjects:
                        filter_code = candidate
                        st.info(f"Detected course code {filter_code}, applying metadata filter to search...")
                
            qa_chain = get_qa_chain(vectorstore, groq_api_key, subject_code=filter_code)
            response = qa_chain.invoke(query)
            
            st.markdown("### Answer")
            st.write(response["answer"])
            
            st.markdown("### Source Citations")
            for i, doc in enumerate(response["context"]):
                subject_code = doc.metadata.get("subject_code", "Unknown")
                page = doc.metadata.get("page", "Unknown")
                with st.expander(f"Source {i+1} (Subject: {subject_code}, Page: {page})"):
                    st.write(doc.page_content)
