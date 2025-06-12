import os
import openai
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def ask_llm(question: str, passages: List[Dict]) -> str:
    """
    Ask the LLM a question using document passages as context.

    Args:
        question: The user's question
        passages: List of relevant chunks with fields like 'text', 'doc_name', 'page'

    Returns:
        Answer string
    """
    try:
        context = "\n\n".join(
            f"{p['doc_name']} (Page {p['page']}):\n{p['text']}" for p in passages
        )

        prompt = f"""You are a helpful assistant. Use the following document content to answer the question accurately and cite the source.

Document Content:
{context}

Question:
{question}

Instructions:
- Use only the information from the documents
- Provide clear and factual answers
- Cite using (Document Name, Page X)
"""

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or "gpt-4" if available
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400
        )

        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"ask_llm failed: {str(e)}")
        return f"Error generating answer: {str(e)}"


def identify_themes(question: str, document_answers: List[Dict]) -> Dict:
    """
    Identify common themes across multiple document answers.

    Args:
        question: The original user question
        document_answers: List of dictionaries containing answers and citations

    Returns:
        A structured dictionary containing:
            - synthesized_answer
            - themes (list)
            - sources (original document_answers)
    """
    try:
        context = "\n\n".join(
            f"DOCUMENT {idx+1} ({ans['doc_name']}, Page {ans['page']}):\n{ans['answer']}"
            for idx, ans in enumerate(document_answers)
        )

        prompt = f"""
Analyze the following answers to a single user question and identify key themes:

Question:
{question}

Answers from different documents:
{context}

Instructions:
1. Identify 1–3 main themes present in the answers.
2. For each theme:
   - Give a short title (3-5 words)
   - Brief description (1-2 sentences)
   - List the DOCUMENT numbers contributing to this theme (e.g., 1, 3, 5)
3. Finally, provide a synthesized answer summarizing all themes.
Format strictly as below:

THEMES:
1. [Theme Name]
   - [Description]
   - Documents: 1, 3

2. [Theme Name]
   - [Description]
   - Documents: 2, 4

SYNTHESIZED ANSWER:
[Summarized and cited answer here]
"""

        response = openai.ChatCompletion.create(
            model="gpt-4",  # Use GPT-4 if enabled
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=600
        )

        return parse_theme_response(response.choices[0].message.content, document_answers)

    except Exception as e:
        logger.error(f"identify_themes failed: {str(e)}")
        return {
            "synthesized_answer": "Theme extraction failed.",
            "themes": [],
            "sources": document_answers
        }


def parse_theme_response(response_text: str, document_answers: List[Dict]) -> Dict:
    """
    Parse the theme extraction response from GPT.

    Args:
        response_text: Raw GPT answer
        document_answers: Original per-document answer list

    Returns:
        Structured response dict with 'themes', 'sources', 'synthesized_answer'
    """
    result = {
        "synthesized_answer": "",
        "themes": [],
        "sources": document_answers
    }

    try:
        # Split between theme section and final answer
        parts = response_text.split("SYNTHESIZED ANSWER:")
        theme_text = parts[0].replace("THEMES:", "").strip() if "THEMES:" in parts[0] else parts[0].strip()
        result["synthesized_answer"] = parts[1].strip() if len(parts) > 1 else "Could not extract answer."

        # Extract individual theme blocks
        blocks = [b.strip() for b in theme_text.split("\n\n") if b.strip()]
        for block in blocks:
            lines = block.split("\n")
            if len(lines) < 3:
                continue

            title = lines[0].lstrip("123.- ").strip()
            description = lines[1].lstrip("- ").strip()
            doc_line = next((l for l in lines if "Documents:" in l), None)

            doc_nums = []
            if doc_line:
                doc_nums = [int(n.strip()) for n in doc_line.split(":")[1].split(",") if n.strip().isdigit()]

            relevant_docs = []
            for n in doc_nums:
                if 1 <= n <= len(document_answers):
                    doc = document_answers[n - 1]
                    relevant_docs.append({
                        "doc_id": doc["doc_id"],
                        "doc_name": doc["doc_name"],
                        "page": doc["page"]
                    })

            result["themes"].append({
                "name": title,
                "description": description,
                "documents": relevant_docs
            })

    except Exception as e:
        logger.error(f"parse_theme_response failed: {str(e)}")

    return result
