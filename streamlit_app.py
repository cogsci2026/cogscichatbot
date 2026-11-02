"""Cogsi – Q&A chatbot for MSc Cognitive Science students (University of Potsdam).

Answers questions using only the documents in docs/text/, via the Mistral API.
Hosted for free on Streamlit Community Cloud. Needs the secret MISTRAL_API_KEY.
"""

import json
import os
import re
from pathlib import Path

import requests
import streamlit as st

MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
DOCS_DIR = Path(__file__).parent / "docs" / "text"

TITLES = {
    "COSEC_FAQ_2024-04-17": "COSEC FAQ (as of 17.04.2024)",
    "COSEC_Study_Regulations_EN": "Study and Examination Regulations, MSc Cognitive Science",
    "BAMA-O_General_Regulations_EN": "General Study and Examination Regulations BA/MA (BAMA-O)",
}


def secret(name, default=None):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, default)


MODEL = secret("MISTRAL_MODEL", "mistral-small-latest")


@st.cache_resource
def system_prompt() -> str:
    parts = []
    for path in sorted(DOCS_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"[ \t]{3,}", "  ", text)  # shrink layout whitespace (saves tokens)
        text = re.sub(r"\n{3,}", "\n\n", text)
        parts.append(f'<document title="{TITLES.get(path.stem, path.stem)}">\n{text}\n</document>')
    docs = "\n\n".join(parts)
    return f"""You are Cogsi, a friendly assistant for students and applicants of the MSc Cognitive Science at the University of Potsdam.

Rules:
- Answer ONLY on the basis of the documents below. Do not use outside knowledge about the university or the program.
- If the documents do not contain the answer, say so clearly and suggest contacting the study advisor or the office named in the documents.
- Name the document (and section or paragraph, e.g. "§ 5") your answer is based on.
- If the program-specific regulations and the general regulations (BAMA-O) differ, the program-specific regulations take precedence; mention this.
- Answer in the language of the question (German or English). Be concise and precise.
- You give information, not legally binding decisions. For binding decisions, refer to the examination board or the responsible office.

Documents:

{docs}
"""


def stream_answer(messages, api_key):
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
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data = line[len("data: "):]
        if data.strip() == "[DONE]":
            break
        delta = json.loads(data)["choices"][0]["delta"].get("content")
        if delta:
            yield delta


st.set_page_config(page_title="Cogsi", page_icon="🧠")
st.title("🧠🎓 Cogsi")
st.caption(
    "Your assistant for the MSc Cognitive Science at the University of Potsdam – "
    "based on the official FAQ and regulations. Fragen gern auch auf Deutsch. "
    "Information only, not legally binding. Please do not enter personal data."
)

api_key = secret("MISTRAL_API_KEY")
if not api_key:
    st.error("MISTRAL_API_KEY is not set. Add it under the app's Settings → Secrets.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar="🧠" if m["role"] == "assistant" else None):
        st.markdown(m["content"])

if question := st.chat_input("Ask a question … / Stell eine Frage …"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    history = st.session_state.messages[-10:]  # keep the last few turns
    messages = [{"role": "system", "content": system_prompt()}] + history
    with st.chat_message("assistant", avatar="🧠"):
        try:
            answer = st.write_stream(stream_answer(messages, api_key))
        except requests.RequestException as e:
            answer = f"⚠️ Connection error: {e}"
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
