import os
import re
import json
import shutil
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

PDF_PATH = "syllabus.pdf"
CHROMA_DB_DIR = "./chroma_db_v5"
SUBJECTS_JSON = "subjects.json"

SEMESTER_HEADERS = {
    "FIRST SEMESTER": "SEM1",
    "SECOND SEMESTER": "SEM2",
    "THIRD SEMESTER": "SEM3",
    "SEMESTER-I": "SEM1",
    "SEMESTER-II": "SEM2",
    "SEMESTER-III": "SEM3",
    "SEMESTER-IV": "SEM4",
    "SEMESTER-V": "SEM5",
    "SEMESTER-VI": "SEM6",
    "SEMESTER-VII": "SEM7",
    "SEMESTER-VIII": "SEM8",
    "1ST SEMESTER": "SEM1",
    "2ND SEMESTER": "SEM2",
    "3RD SEMESTER": "SEM3",
    "4TH SEMESTER": "SEM4",
    "5TH SEMESTER": "SEM5",
    "6TH SEMESTER": "SEM6",
    "7TH SEMESTER": "SEM7",
    "8TH SEMESTER": "SEM8",
    "7THSEMESTER": "SEM7",
    "8THSEMESTER": "SEM8",
}

def extract_text_and_metadata(pdf_path):
    print(f"Reading {pdf_path}...")
    
    subjects = {}
    
    current_code = "GENERAL"
    current_title = "General Regulations and Curriculum"
    
    subjects[current_code] = {"title": current_title, "text": "", "page_map": []}
    
    header_pattern = re.compile(r'^([A-Z]{2,4}\s*\d{3})\s+(.*?(?:Credits?|Hours?|PCR|PEL|Elective).*?)$', re.IGNORECASE)
    fallback_pattern = re.compile(r'^([A-Z]{2,4}\s*\d{3})\s+([A-Za-z\s\-]+)$')
    
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_num = i + 1
            text = page.extract_text()
            if not text:
                continue
            
            lines = text.split('\n')
            
            for line in lines:
                line_upper = line.strip().upper()
                sem_match = None
                for header, sem_code in SEMESTER_HEADERS.items():
                    if line_upper == header:
                        sem_match = sem_code
                        break
                
                if sem_match:
                    code = sem_match
                    title = f"Semester {code.replace('SEM', '')} Course List"
                    if code not in subjects:
                        subjects[code] = {"title": title, "text": "", "page_map": []}
                    current_code = code
                    current_title = title
                    # Proceed to append this line to the block text
                else:
                    match = header_pattern.match(line)
                    if not match:
                        match = fallback_pattern.match(line)
                        
                    # Prevent table rows (containing L T S C H digit sequences or POs) from being classified as headers
                    if match and (re.search(r'\d\s+\d\s+\d', line) or re.search(r'\b[CP]O\d+\b', line)):
                        match = None

                    if match:
                        code = match.group(1).replace(" ", "") 
                        title = match.group(2).strip()
                        
                        if code not in subjects:
                            subjects[code] = {"title": title, "text": "", "page_map": []}
                        current_code = code
                        current_title = title
                        
                char_offset = len(subjects[current_code]["text"])
                if not subjects[current_code]["page_map"] or subjects[current_code]["page_map"][-1][1] != page_num:
                    subjects[current_code]["page_map"].append((char_offset, page_num))
                
                subjects[current_code]["text"] += line + "\n"
                
                # Once we hit the 'TOTAL' line for a semester, the table is finished!
                # Revert back to the GENERAL bucket so we don't accidentally swallow the electives or index.
                if current_code.startswith("SEM") and line.strip().upper().startswith("TOTAL"):
                    current_code = "GENERAL"
                    current_title = "General Regulations and Curriculum"

    print(f"Identified {len(subjects)} unique subject blocks.")
    
    with open(SUBJECTS_JSON, "w") as f:
        json.dump(list(subjects.keys()), f)
        
    return subjects

def get_page_for_offset(offset, page_map):
    current_page = page_map[0][1] if page_map else "Unknown"
    for map_offset, page_num in page_map:
        if offset >= map_offset:
            current_page = page_num
        else:
            break
    return current_page

def main():
    print("Extracting and parsing PDF...")
    subjects = extract_text_and_metadata(PDF_PATH)
    
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True
    )
    
    documents = []
    
    for code, data in subjects.items():
        title = data["title"]
        text = data["text"]
        page_map = data["page_map"]
        
        chunks = text_splitter.create_documents([text])
        
        for chunk in chunks:
            chunk_offset = chunk.metadata.get("start_index", 0)
            page_num = get_page_for_offset(chunk_offset, page_map)
            
            enriched_content = f"[{code} - {title}]\n{chunk.page_content}"
            
            doc = Document(
                page_content=enriched_content,
                metadata={
                    "page": page_num,
                    "subject_code": code,
                    "source": PDF_PATH
                }
            )
            documents.append(doc)
            
    print(f"Created {len(documents)} context-enriched chunks.")
    
    print("Initializing embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if os.path.exists(CHROMA_DB_DIR):
        print(f"Cleaning existing vector store at {CHROMA_DB_DIR}...")
        shutil.rmtree(CHROMA_DB_DIR)
        
    print(f"Creating ChromaDB vector store at {CHROMA_DB_DIR}...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    
    print("Ingestion complete! Vector store saved successfully.")

if __name__ == "__main__":
    main()
