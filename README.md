# Cogsi – CogSci Chatbot

Chatbot for students and applicants of the MSc Cognitive Science (University of Potsdam). Answers questions using only the official documents in `docs/`, via the Mistral API. Hosted for free on Streamlit Community Cloud.

## Documents

| File | Content |
|---|---|
| `docs/pdf/COSEC_FAQ_2024-04-17.pdf` | COSEC FAQ (as of 17.04.2024) |
| `docs/pdf/COSEC_Study_Regulations_EN.pdf` | Study and examination regulations, MSc Cognitive Science (EN) |
| `docs/pdf/BAMA-O_General_Regulations_EN.pdf` | General study and examination regulations for BA/MA programs (BAMA-O, EN) |

The bot reads the plain-text versions in `docs/text/`. To add a document: put the PDF in `docs/pdf/`, run `pdftotext -layout file.pdf docs/text/file.txt`, and optionally add a title in `TITLES` in `streamlit_app.py`.

## Setup (Streamlit Community Cloud, free)

1. Sign in at https://share.streamlit.io with GitHub.
2. Create app → repository `cogsci2026/cogscichatbot`, branch `main`, file `streamlit_app.py`.
3. Under Advanced settings → Secrets, add:
   ```toml
   MISTRAL_API_KEY = "your-key"
   ```
4. Deploy. Every push to `main` updates the app automatically.

Optional secret `MISTRAL_MODEL` (default: `mistral-small-latest`).

## Run locally

```bash
pip install -r requirements.txt
MISTRAL_API_KEY=... streamlit run streamlit_app.py
```
