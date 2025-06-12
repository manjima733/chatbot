import streamlit as st
import requests
import os
import time

# === CONFIG ===
BACKEND_URL = "http://localhost:8000"
MAX_FILE_SIZE_MB = 200
ALLOWED_FILE_TYPES = ["pdf", "png", "jpg", "jpeg", "txt"]

# === STATE INIT ===
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# === PAGE SETUP ===
st.set_page_config(page_title=" Gen-AI Document Q&A", layout="wide")
st.title(" Gen-AI Document Q&A Chatbot")
st.caption("Upload documents and ask questions about their content")

# === SIDEBAR ===
with st.sidebar:
    st.title("Document Management")

    uploaded_file = st.file_uploader(
        "Upload documents (PDF, Images, Text)",
        type=ALLOWED_FILE_TYPES,
        accept_multiple_files=False
    )

    if uploaded_file:
        # Avoid re-upload
        already_uploaded = any(f["name"] == uploaded_file.name for f in st.session_state.uploaded_files)
        if not already_uploaded:
            file_size = uploaded_file.size / (1024 * 1024)
            if file_size > MAX_FILE_SIZE_MB:
                st.error(f"File too large. Max size: {MAX_FILE_SIZE_MB}MB")
            else:
                with st.spinner(f"Uploading {uploaded_file.name}..."):
                    try:
                        res = requests.post(
                            f"{BACKEND_URL}/upload/",
                            files={"file": uploaded_file}
                        )
                        if res.status_code == 200:
                            st.success(" Upload successful!")
                            st.session_state.uploaded_files.append({
                                "name": uploaded_file.name,
                                "size": f"{file_size:.2f}MB"
                            })
                            st.rerun()
                        else:
                            st.error(f" Upload failed: {res.text}")
                    except requests.exceptions.ConnectionError:
                        st.error(" Could not connect to backend")
        else:
            st.info(" This file is already uploaded.")

    if st.session_state.uploaded_files:
        st.divider()
        st.subheader("Uploaded Documents")
        for file in st.session_state.uploaded_files:
            st.markdown(f" {file['name']} ({file['size']})")

        if st.button("🔄 Clear All Documents"):
            st.session_state.uploaded_files = []
            st.rerun()

# === TABS ===
tab1, tab2 = st.tabs([" Chat", " Document Viewer"])

# === CHAT TAB ===
with tab1:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                with st.expander(" View Sources"):
                    for source in message["sources"]:
                        st.markdown(f"**{source['doc_name']}** (page {source['page']})")
                        st.caption(source['para'][:200] + "...")
                        st.divider()

    # Question Input
    if prompt := st.chat_input("Ask a question about your documents..."):
        if not st.session_state.uploaded_files:
            st.warning("Please upload at least one document first")
            st.stop()

        # Add user input to chat
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing documents..."):
                try:
                    start = time.time()
                    response = requests.post(
                        f"{BACKEND_URL}/ask/",
                        json={"question": prompt}
                    )
                    if response.status_code == 200:
                        data = response.json()
                        st.markdown(data["synthesized_answer"])

                        if "themes" in data:
                            with st.expander(" Identified Themes"):
                                for theme in data["themes"]:
                                    st.markdown(f"**{theme['name']}**")
                                    st.caption(theme["description"])
                                    for doc in theme["documents"]:
                                        st.markdown(f"- {doc['doc_name']} (Page {doc['page']})")

                        st.caption(f"️ Response time: {time.time() - start:.2f}s")

                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": data["synthesized_answer"],
                            "sources": data.get("sources", []),
                            "themes": data.get("themes", [])
                        })
                    else:
                        st.error(" Error from backend: " + response.text)
                except requests.exceptions.RequestException:
                    st.error(" Could not connect to the backend server")

# === DOCUMENT VIEWER TAB ===
with tab2:
    if st.session_state.uploaded_files:
        st.subheader("Document Explorer")
        selected = st.selectbox(
            "Select a document",
            options=[f["name"] for f in st.session_state.uploaded_files]
        )
        st.warning(" Document viewer not yet implemented.")
    else:
        st.info("Upload documents to view them here.")

# === FOOTER ===
st.divider()
st.caption("Built with Streamlit and FastAPI | Wasserstoff Gen-AI Internship Task")
