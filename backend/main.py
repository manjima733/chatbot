from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import os
from fastapi.middleware.cors import CORSMiddleware
from backend.ocr_utils import ocr_processor
from backend.vector_utils import vector_store
from backend.qa_utils import ask_llm, identify_themes

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "backend/storage/uploaded"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QuestionInput(BaseModel):
    question: str

@app.get("/ping")
def ping():
    return {"message": "pong"}

@app.post("/upload/")
async def upload_doc(file: UploadFile):
    try:
        print(f"📥 Received file: {file.filename}")

        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        print("✅ File saved. Starting OCR...")
        text = ocr_processor.extract_text(file_path)
        print(f" Extracted {len(text)} characters.")
        print(" Extracted Text Preview:")
        print(text[:1000])

        if not text.strip():
            print("❌ No text extracted from document.")
            return {"error": "No text extracted from the uploaded document."}

        doc_id = file.filename
        print(" Calling vector_store.add_document...")

        from backend.vector_utils import VectorStore
        if isinstance(vector_store, VectorStore):
            paragraphs = vector_store._split_text(text)
            print(f" Raw OCR text sample:\n{text[:500]}")
            print(f" Paragraph count: {len(paragraphs)}")
            print(f" Sample paragraphs: {paragraphs[:2]}")

        success = vector_store.add_document(text, doc_id=doc_id, doc_name=file.filename)

        if not success:
            print(" Vector store rejected the document. Possibly no valid text to embed.")
            return {"error": "No valid content found in the document to process."}

        print(f" Document '{doc_id}' indexed with vector count: {vector_store.index.ntotal}")
        return {"message": "✅ Upload and processing complete."}

    except Exception as e:
        print(f"[UPLOAD ERROR] {str(e)}")
        return {
            "error": f"Upload failed: {str(e)}"
        }

@app.post("/ask/")
async def ask_question(data: QuestionInput):
    try:
        question = data.question
        print("==== DEBUGGING SEARCH ====")
        print(f"[ASK] Question: {question}")
        print(f"Total vectors in index: {vector_store.index.ntotal}")
        print(f"Uploaded documents: {list(vector_store.document_metadata.keys())}")
        print(f"Chunks stored: {len(vector_store.chunks)}")
        print("===========================")

        results = vector_store.search(question, top_k=5)
        print(f" Vector search results: {len(results)} chunks")

        if not results:
            return {
                "synthesized_answer": "️ No relevant content found.",
                "themes": [],
                "sources": []
            }

        document_answers = []
        for chunk in results:
            answer = ask_llm(question, [chunk])
            document_answers.append({
                "doc_id": chunk.get("doc_id", "unknown"),
                "doc_name": chunk.get("doc_name", "unknown"),
                "page": chunk.get("page", 1),
                "para": chunk.get("text", ""),
                "answer": answer
            })

        theme_response = identify_themes(question, document_answers)

        if "synthesized_answer" not in theme_response:
            theme_response["synthesized_answer"] = "️Theme synthesis failed."
            theme_response.setdefault("themes", [])
            theme_response.setdefault("sources", document_answers)

        return theme_response

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return {
            "synthesized_answer": "️ Failed to process your question.",
            "themes": [],
            "sources": [],
            "error": str(e)
        }
