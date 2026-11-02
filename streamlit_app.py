"""Cogsi – Q&A chatbot for MSc Cognitive Science students (University of Potsdam).

Answers questions using only the documents in docs/text/, via the Mistral API.
Hosted for free on Streamlit Community Cloud. Needs the secret MISTRAL_API_KEY.
"""

import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import requests
import streamlit as st

MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
EMBED_URL = "https://api.mistral.ai/v1/embeddings"
TOP_K = 8  # number of text chunks sent per question
DOCS_DIR = Path(__file__).parent / "docs" / "text"
AVATAR = str(Path(__file__).parent / "assets" / "cogsi.png")

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


def post_with_retry(url, api_key, payload, stream=False, tries=4):
    """POST to Mistral; on rate limit (429) wait and retry."""
    for i in range(tries):
        r = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            stream=stream,
            timeout=120,
        )
        if r.status_code != 429 or i == tries - 1:
            return r
        time.sleep(2 * (i + 1) ** 2)  # 2, 8, 18 s
    return r


@st.cache_resource
def load_chunks():
    """Split the documents into ~1500-character chunks."""
    chunks = []
    for path in sorted(DOCS_DIR.glob("*.txt")):
        title = TITLES.get(path.stem, path.stem)
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"[ \t]{3,}", "  ", text)
        paras = []
        for p in re.split(r"\n\s*\n", text):
            lines, part = p.strip().splitlines(), ""
            for line in lines:  # split very long paragraphs at line breaks
                if part and len(part) + len(line) > 1200:
                    paras.append(part)
                    part = ""
                part += line + "\n"
            if part.strip():
                paras.append(part.strip())
        buf = ""
        for para in paras:
            if buf and len(buf) + len(para) > 1500:
                chunks.append({"title": title, "text": buf})
                buf = buf[-200:]  # small overlap
            buf += "\n\n" + para
        if buf.strip():
            chunks.append({"title": title, "text": buf})
    return chunks


def tokens(text):
    return re.findall(r"\w+", text.lower())


@st.cache_resource(show_spinner="Preparing documents …")
def chunk_embeddings(api_key):
    """Embed all chunks once (multilingual, so German questions match English text)."""
    texts = [f"{c['title']}\n{c['text']}" for c in load_chunks()]
    vecs = []
    for i in range(0, len(texts), 32):
        r = post_with_retry(EMBED_URL, api_key, {"model": "mistral-embed", "input": texts[i:i + 32]})
        r.raise_for_status()
        vecs += [d["embedding"] for d in r.json()["data"]]
        time.sleep(1.1)  # stay under the free tier's requests-per-second limit
    m = np.array(vecs, dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def bm25_ranking(query):
    """Keyword fallback if embeddings are unavailable."""
    docs = [Counter(tokens(c["text"])) for c in load_chunks()]
    avg = sum(sum(d.values()) for d in docs) / len(docs)
    df = Counter(t for d in docs for t in d)
    scores = []
    for d in docs:
        dl = sum(d.values())
        sc = 0.0
        for t in set(tokens(query)):
            if t in d:
                idf = math.log(1 + (len(docs) - df[t] + 0.5) / (df[t] + 0.5))
                sc += idf * d[t] * 2.2 / (d[t] + 1.2 * (0.25 + 0.75 * dl / avg))
        scores.append(sc)
    return np.argsort(scores)[::-1]


def retrieve(query, api_key):
    chunks = load_chunks()
    try:
        m = chunk_embeddings(api_key)
        r = post_with_retry(EMBED_URL, api_key, {"model": "mistral-embed", "input": [query]})
        r.raise_for_status()
        q = np.array(r.json()["data"][0]["embedding"], dtype=np.float32)
        order = np.argsort(m @ (q / np.linalg.norm(q)))[::-1]
    except Exception:
        order = bm25_ranking(query)
    picked = sorted(order[:TOP_K])  # keep document order
    return "\n\n".join(
        f'<excerpt document="{chunks[i]["title"]}">\n{chunks[i]["text"].strip()}\n</excerpt>' for i in picked
    )


SYSTEM_PROMPT = """You are Cogsi, a friendly assistant for students and applicants of the MSc Cognitive Science at the University of Potsdam.

Rules:
- Answer ONLY on the basis of the document excerpts provided with the question. Do not use outside knowledge about the university or the program.
- If the excerpts do not contain the answer, say so clearly and suggest contacting the study advisor or the office named in the documents.
- Name the document (and section or paragraph, e.g. "§ 5") your answer is based on.
- If the program-specific regulations and the general regulations (BAMA-O) differ, the program-specific regulations take precedence; mention this.
- Answer in the language of the question (German or English). Be concise and precise.
- You give information, not legally binding decisions. For binding decisions, refer to the examination board or the responsible office."""


def stream_answer(messages, api_key):
    r = post_with_retry(
        MISTRAL_URL, api_key,
        {"model": MODEL, "messages": messages, "temperature": 0.2, "stream": True},
        stream=True,
    )
    if r.status_code == 429:
        yield "⏳ Too many questions at the moment – please wait a minute and try again. / Gerade zu viele Anfragen – bitte in einer Minute nochmal versuchen."
        return
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


st.set_page_config(page_title="Cogsi", page_icon=AVATAR)
col_img, col_title = st.columns([1, 6], vertical_alignment="center")
col_img.image(AVATAR, width=80)
col_title.title("Cogsi")
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
    with st.chat_message(m["role"], avatar=AVATAR if m["role"] == "assistant" else None):
        st.markdown(m["content"])

if question := st.chat_input("Ask a question … / Stell eine Frage …"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    history = st.session_state.messages[-7:-1]  # previous turns (without the new question)
    prev_q = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    with st.chat_message("assistant", avatar=AVATAR):
        try:
            with st.spinner("Searching the documents …"):
                context = retrieve(f"{prev_q}\n{question}".strip(), api_key)
            messages = (
                [{"role": "system", "content": SYSTEM_PROMPT}]
                + history
                + [{"role": "user", "content": f"Document excerpts:\n\n{context}\n\nQuestion: {question}"}]
            )
            answer = st.write_stream(stream_answer(messages, api_key))
        except requests.RequestException as e:
            answer = f"⚠️ Connection error: {e}"
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
