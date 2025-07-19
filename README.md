What it does
- Lets users upload many kinds of documents — PDFs, scanned files, images, or plain text
- Extracts text using OCR if needed (for scanned files or images)
- Breaks the content into paragraphs, converts them into embeddings (vectors)
- Stores them in FAISS 
- Lets the user ask natural language questions
- Searches for the most relevant chunks and asks GPT for an answer
- Then summarizes and identifies key themes across all document answers
- Displays everything — answers, sources, themes — in a clean and easy-to-use web UI


Tech Stack
-FastAPI for the backend: I used FastAPI to handle things like uploading files and getting answers from the chatbot. It’s fast and easy to work with.
-Streamlit for the frontend:This is the part the user sees. I chose Streamlit because it made it easy to build a simple and clean website where users can upload files and ask questions.
-Tesseract and PyMuPDF for reading documents:These tools help the system read files. PyMuPDF reads normal PDF text, and if the file is a scanned image or photo, Tesseract helps read it using OCR (optical character recognition), like reading text from pictures.
-FAISS and MiniLM for finding answers:After the documents are read, it breaks them into smaller parts and use MiniLM to understand their meaning. Then FAISS helps quickly find the parts most related to the question asked.
-OpenAI GPT-3.5 / GPT-4 for smart answers:I used GPT to read the best parts of the documents and give answers in a clear way — and even find common ideas or themes from multiple documents.


How it works
1.Upload a document
- You can upload PDFs, images, or `.txt` files.
- If it’s scanned, I used Tesseract OCR to extract the text.
- Everything is saved and logged.

2.Text gets embedded
- Using a transformer model (MiniLM), I turn each paragraph into a vector.
- These go into a FAISS index for fast semantic search.
- I also store metadata like page number, document name, etc.

3. Ask a question
- When you type a question, I embed it the same way.
- I search FAISS for the most relevant chunks.
- Then I send those chunks +  question to OpenAI’s GPT.

4. See the answer (with sources)
- GPT answers the question based only on the relevant chunks.
- I cite document name, page, and paragraph in the response.

 5. Find the themes
- I take all document-level answers and send them again to GPT.
- It returns the top **themes** it finds, with a summary and source list.
- All this is shown in the Streamlit app.

Project Structure
chatbot_theme_identifier/
├── backend/
│ ├── main.py ← FastAPI server
│ ├── ocr_utils.py ← OCR and text extraction
│ ├── vector_utils.py ← Embedding + FAISS storage
│ ├── qa_utils.py ← GPT Q&A and theme summarization
│ └── storage/ ← Saved docs, embeddings, metadata
| |──.env
|── frontend/
│ └── app.py ← Streamlit interface
├── requirements.txt


Start backend
 uvicorn backend.main:app --reload --port 8000

Start frontend
python -m streamlit run frontend/app.py 
