import os
import yaml
import json
import re
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_qa_chain(vectorstore, groq_api_key, subject_code=None):
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
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
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    rag_chain_from_docs = (RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"]))) | prompt | llm | StrOutputParser())
    return RunnableParallel({"context": retriever, "input": RunnablePassthrough()}).assign(answer=rag_chain_from_docs)

def judge_answer(groq_api_key, question, expected, actual):
    if expected == "REFUSAL":
        return "I am sorry, but I cannot find the answer" in actual
        
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an evaluator. Compare the ACTUAL answer to the EXPECTED answer for the given question. "
                   "If the ACTUAL answer correctly captures the EXPECTED answer, output exactly 'YES'. "
                   "Otherwise output exactly 'NO'. Do not output anything else."),
        ("human", "Question: {q}\nExpected: {e}\nActual: {a}")
    ])
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"q": question, "e": expected, "a": actual})
    return "YES" in result.upper()

def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Please set GROQ_API_KEY environment variable. Run with: set GROQ_API_KEY=your_key && python eval.py")
        return
        
    if not api_key.isascii():
        print("ERROR: Your GROQ_API_KEY contains invalid characters (e.g., an em-dash). Please ensure it is pasted cleanly.")
        return
        
    api_key = api_key.strip()
        
    print("Loading testset...")
    with open("eval/testset.yaml", "r") as f:
        testset = yaml.safe_load(f)
        
    print("Loading known subjects...")
    try:
        with open("subjects.json", "r") as f:
            known_subjects = set(json.load(f))
    except Exception:
        known_subjects = set()
        
    print("Loading vectorstore...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="./chroma_db_v4", embedding_function=embeddings)
    
    total = len(testset)
    hit_5 = 0
    hit_5_total = 0
    answer_match = 0
    adversarial_accuracy = 0
    adversarial_total = 0
    
    print(f"Running evaluation on {total} questions...\n")
    
    with open("eval_results.md", "w") as f:
        f.write("# RAG Evaluation Results\n\n")
        f.write("| Question | Expected | Hit@5 | Answer Match |\n")
        f.write("|---|---|---|---|\n")
        
        for item in testset:
            q = item['question']
            expected = item['expected_answer']
            subject_target = item.get('subject_code')
            q_type = item['type']
            
            subject_match = re.search(r'\b([a-zA-Z]{2,4})\s*(\d{2,3})\b', q)
            filter_code = None
            if subject_match:
                candidate = (subject_match.group(1) + subject_match.group(2)).upper()
                if candidate in known_subjects:
                    filter_code = candidate
            
            chain = get_qa_chain(vectorstore, api_key, filter_code)
            res = chain.invoke(q)
            
            retrieved_subjects = [doc.metadata.get("subject_code") for doc in res["context"]]
            
            hit = False
            if subject_target and q_type == "single_subject":
                hit_5_total += 1
                if subject_target in retrieved_subjects:
                    hit = True
                    hit_5 += 1
            
            match = judge_answer(api_key, q, expected, res["answer"])
            if match:
                answer_match += 1
                
            if q_type == "adversarial":
                adversarial_total += 1
                if match:
                    adversarial_accuracy += 1
                    
            f.write(f"| {q} | {expected} | {'Yes' if hit else ('N/A' if not subject_target or q_type != 'single_subject' else 'No')} | {'Yes' if match else 'No'} |\n")
            print(f"Q: {q}\nMatch: {match}, Hit@5: {hit}\n---")
            
        f.write("\n## Summary Metrics\n")
        f.write(f"- **Hit@5 (Single Subject)**: {hit_5}/{hit_5_total} ({hit_5/hit_5_total*100:.1f}%)\n" if hit_5_total > 0 else "")
        f.write(f"- **Answer Match Rate**: {answer_match}/{total} ({answer_match/total*100:.1f}%)\n")
        f.write(f"- **Adversarial Accuracy**: {adversarial_accuracy}/{adversarial_total} ({adversarial_accuracy/adversarial_total*100:.1f}%)\n" if adversarial_total > 0 else "")

    print(f"\nEvaluation complete. Results saved to eval_results.md.")

if __name__ == "__main__":
    main()
