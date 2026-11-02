"""Cogsi – Q&A chatbot for MSc Cognitive Science students (University of Potsdam).

Answers questions using only the documents in docs/text/, via the Mistral API.
Requires the environment variable / Space secret MISTRAL_API_KEY.
"""

import json
import os
import re
from pathlib import Path

import gradio as gr
import requests

MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
MODEL = os.environ.get("MISTRAL_MODEL", "mistral-small-latest")
DOCS_DIR = Path(__file__).parent / "docs" / "text"

# Human-readable titles for the documents (file stem -> title)
TITLES = {
    "COSEC_FAQ_2024-04-17": "COSEC FAQ (as of 17.04.2024)",
    "COSEC_Study_Regulations_EN": "Study and Examination Regulations, MSc Cognitive Science",
    "BAMA-O_General_Regulations_EN": "General Study and Examination Regulations BA/MA (BAMA-O)",
}


def load_documents() -> str:
    parts = []
    for path in sorted(DOCS_DIR.glob("*.txt")):
        title = TITLES.get(path.stem, path.stem)
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"[ \t]{3,}", "  ", text)  # shrink layout whitespace (saves tokens)
        text = re.sub(r"\n{3,}", "\n\n", text)
        parts.append(f'<document title="{title}">\n{text}\n</document>')
    return "\n\n".join(parts)


DOCUMENTS = load_documents()

SYSTEM_PROMPT = f"""You are Cogsi, a friendly assistant for students and applicants of the MSc Cognitive Science at the University of Potsdam.

Rules:
- Answer ONLY on the basis of the documents below. Do not use outside knowledge about the university or the program.
- If the documents do not contain the answer, say so clearly and suggest contacting the study advisor or the office named in the documents.
- Name the document (and section or paragraph, e.g. "§ 5") your answer is based on.
- If the program-specific regulations and the general regulations (BAMA-O) differ, the program-specific regulations take precedence; mention this.
- Answer in the language of the question (German or English). Be concise and precise.
- You give information, not legally binding decisions. For binding decisions, refer to the examination board or the responsible office.

Documents:

{DOCUMENTS}
"""


def to_messages(history):
    """Convert Gradio history (messages or tuple format) to API messages."""
    msgs = []
    for item in history:
        if isinstance(item, dict):
            content = item.get("content")
            if isinstance(content, str) and item.get("role") in ("user", "assistant"):
                msgs.append({"role": item["role"], "content": content})
        else:
            user, bot = item
            if user:
                msgs.append({"role": "user", "content": user})
            if bot:
                msgs.append({"role": "assistant", "content": bot})
    return msgs


def respond(message, history):
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        yield "⚠️ MISTRAL_API_KEY is not set. Please add it as a secret in the Space settings."
        return

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += to_messages(history)[-10:]  # keep the last few turns
    messages.append({"role": "user", "content": message})

    try:
        r = requests.post(
            MISTRAL_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": MODEL, "messages": messages, "temperature": 0.2, "stream": True},
            stream=True,
            timeout=120,
        )
        if r.status_code != 200:
            yield f"⚠️ Error from Mistral API ({r.status_code}): {r.text[:300]}"
            return

        answer = ""
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data.strip() == "[DONE]":
                break
            chunk = json.loads(data)
            delta = chunk["choices"][0]["delta"].get("content") or ""
            answer += delta
            yield answer
    except requests.RequestException as e:
        yield f"⚠️ Connection error: {e}"


demo = gr.ChatInterface(
    fn=respond,
    type="messages",
    title="🧠🎓 Cogsi – your MSc Cognitive Science assistant",
    description=(
        "Ask me about the MSc Cognitive Science at the University of Potsdam – "
        "application, study regulations, exams, modules. I answer based on the official FAQ "
        "and regulations. Fragen gern auch auf Deutsch.<br>"
        "<small>Information only – not legally binding. Please do not enter personal data.</small>"
    ),
    examples=[
        "How many credit points do I need for the Master's thesis?",
        "Can I study part-time?",
        "Wie oft darf ich eine Prüfung wiederholen?",
    ],
)

if __name__ == "__main__":
    demo.launch()
