"""Cogsi – Q&A chatbot for MSc Cognitive Science students (University of Potsdam).

Answers questions using only the documents in docs/text/, via the Mistral API.
Hosted for free on Streamlit Community Cloud. Needs the secret MISTRAL_API_KEY.
"""

import base64
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
TOP_K = 12  # number of text chunks sent per question
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


STOP = set("the a an of to in and or for on at by is are be with as this that from it its any".split())


def tokens(text):
    out = []
    for w in re.findall(r"\w+", text.lower()):
        if w in STOP or len(w) < 2:
            continue
        for suf in ("ations", "ation", "ings", "ing", "ies", "es", "s"):
            if len(w) > len(suf) + 3 and w.endswith(suf):
                w = w[: -len(suf)]
                break
        out.append(w)
    return out


def bm25_ranking(query):
    """Rank chunks by keyword relevance (BM25)."""
    docs = [Counter(tokens(c["title"] + " " + c["text"])) for c in load_chunks()]
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


def search_keywords(question, prev_q, api_key):
    """Turn the question (any language) into English search keywords with one small, cheap call."""
    prompt = (
        "Convert this student question about a university master's program into 10-20 English search "
        "keywords for searching study and examination regulations. Include synonyms "
        "(e.g. exam, examination, retake, repeat, credit points, CP, thesis). Output only the keywords.\n\n"
        + (f"Previous question: {prev_q}\n" if prev_q else "")
        + f"Question: {question}"
    )
    try:
        r = post_with_retry(MISTRAL_URL, api_key, {
            "model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": 80,
        })
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception:
        return ""


def retrieve(question, prev_q, api_key):
    chunks = load_chunks()
    keywords = search_keywords(question, prev_q, api_key)
    time.sleep(1.1)  # free tier: about one request per second
    order = bm25_ranking(f"{question} {keywords}")
    picked = sorted(order[:TOP_K])  # keep document order
    return "\n\n".join(
        f'<excerpt document="{chunks[i]["title"]}">\n{chunks[i]["text"].strip()}\n</excerpt>' for i in picked
    )


SYSTEM_PROMPT = """You are Cogsi, a friendly assistant for students and applicants of the MSc Cognitive Science at the University of Potsdam.

Rules:
- Answer ONLY on the basis of the document excerpts provided with the question. Do not use outside knowledge about the university or the program.
- If the excerpts do not contain the answer, say clearly that you don't know and suggest contacting the study advisor or the office named in the documents.
- Do NOT mention or cite the documents, excerpts, sections or paragraphs in your answer (no "according to the FAQ", no "§ 5"). Just give the answer directly, as a knowledgeable study advisor would.
- If the program-specific regulations and the general regulations (BAMA-O) differ, the program-specific regulations take precedence.
- Answer in the language of the question (German or English).
- Give complete, detailed answers: include all relevant conditions, numbers, deadlines, exceptions and practical next steps (e.g. whom to contact, what to submit). Use short paragraphs or bullet points for readability. If several rules apply, explain each.
- You give information, not legally binding decisions. For binding decisions, refer to the examination board or the responsible office."""


def stream_answer(messages, api_key):
    r = post_with_retry(
        MISTRAL_URL, api_key,
        {"model": MODEL, "messages": messages, "temperature": 0.2, "stream": True},
        stream=True,
    )
    if r.status_code == 429:
        limits = {k: v for k, v in r.headers.items() if "ratelimit" in k.lower() or k.lower() == "retry-after"}
        yield (
            "⏳ Too many questions at the moment – please wait a minute and try again.\n\n"
            f"<small>Details: model `{MODEL}` · {r.text[:200]} · {limits}</small>"
        )
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
_logo = base64.b64encode(Path(AVATAR).read_bytes()).decode()
st.markdown(
    f'<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.5rem">'
    f'<img src="data:image/png;base64,{_logo}" width="72">'
    f'<span style="font-size:2.6rem;font-weight:700">Cogsi</span></div>',
    unsafe_allow_html=True,
)
st.caption(
    "Your assistant for the MSc Cognitive Science at the University of Potsdam – "
    "based on the official FAQ and regulations. "
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

if question := st.chat_input("Ask a question …"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    history = st.session_state.messages[-7:-1]  # previous turns (without the new question)
    prev_q = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    with st.chat_message("assistant", avatar=AVATAR):
        try:
            with st.spinner("Searching the documents …"):
                context = retrieve(question, prev_q, api_key)
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
