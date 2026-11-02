---
title: Cogsi
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
---

# Cogsi – CogSci Chatbot

Chatbot for students and applicants of the MSc Cognitive Science (University of Potsdam). Answers questions using only the official documents in `docs/`, via the Mistral API.

## Documents

| File | Content |
|---|---|
| `docs/pdf/COSEC_FAQ_2024-04-17.pdf` | COSEC FAQ (as of 17.04.2024) |
| `docs/pdf/COSEC_Study_Regulations_EN.pdf` | Study and examination regulations, MSc Cognitive Science (EN) |
| `docs/pdf/BAMA-O_General_Regulations_EN.pdf` | General study and examination regulations for BA/MA programs (BAMA-O, EN) |

The bot reads the plain-text versions in `docs/text/`. To add a document: put the PDF in `docs/pdf/`, run `pdftotext -layout file.pdf docs/text/file.txt`, and optionally add a title in `TITLES` in `app.py`.

## Setup

- **Hugging Face Space** (`cogsciup/guidelines`): add the secret `MISTRAL_API_KEY` under Settings → Variables and secrets.
- **Auto-deploy from GitHub**: add a Hugging Face write token as the GitHub repo secret `HF_TOKEN`. Every push to `main` then updates the Space (PDFs are left out; the Space only needs the text files).
- Optional: set the variable `MISTRAL_MODEL` to use another model (default: `mistral-small-latest`).

## Run locally

```bash
pip install gradio==5.49.1 requests
MISTRAL_API_KEY=... python app.py
```
