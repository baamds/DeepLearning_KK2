
# Vårdcentralen Solrosen – RAG-assistent (lokal via Ollama)

#   Läser in PDF-dokument i ./docs/
#   Delar upp dem i "chunks" (textbitar) och bygger ett FAISS-index
#   Hämtar relevanta bitar och låter en lokal LLM (Ollama)
#   generera ett svar som STRIKT ska hålla sig till den hämtade kontexten


import os
import io
import textwrap
from pathlib import Path
from typing import List, Dict

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
from pypdf import PdfReader
import ollama  # Python-klient som pratar med lokala Ollama (http://localhost:11434)
from tqdm import tqdm


# Grundinställningar

APP_TITLE = "Vårdcentralen Solrosen – RAG (Lokal Ollama)"
DOCS_DIR = Path("docs")               # Här har PDF:erna (källor)
EMBED_MODEL_NAME = "all-MiniLM-L6-v2" # Litet & snabbt embedding-modellnamn
OLLAMA_MODEL_DEFAULT = "llama3:instruct"
TOP_K = 3                             # Hur många textbitar hämtas per fråga
CHUNK_SIZE = 500                      # Max tecken per chunk
CHUNK_OVERLAP = 150                   # Överlapp för bättre kontext

# Streamlit-sidlayout
st.set_page_config(page_title=APP_TITLE, page_icon="🩺", layout="wide")
st.title(APP_TITLE)
st.caption("⚠️ Generell informationsassistent. Ersätter inte medicinsk bedömning. Vid akuta besvär: ring 112.")


# Hjälpfunktion: läsa text från PDF (pypdf)

def read_pdf_text(pdf_path: Path) -> str:
    """
    Läser ut all text från en PDF med pypdf.
    Om någon sida saknar extraherbar text returnerar vi tom sträng där.
    """
    text_parts = []
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            # extract_text() kan returnera None om sidan är "skannad" utan OCR.
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


# Hjälpfunktion: skapa textbitar (chunks) med overlap

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Delar upp text i överlappande bitar (teckenbaserat).
    """
    text = (text or "").replace("\r", "")
    if not text.strip():
        return []

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


# Bygg korpus från PDF:er i DOCS_DIR

def build_corpus_from_pdfs() -> List[Dict]:
    """
    Går igenom DOCS_DIR och bygger en lista av {text, source, chunk_id}.
    Endast PDF-filer behandlas i denna lösning (enligt önskemål).
    """
    DOCS_DIR.mkdir(exist_ok=True)
    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    corpus = []
    if not pdfs:
        st.warning("Hittade inga PDF-filer i ./docs/. Ladda upp PDF:er i sidomenyn eller lägg dem i mappen och klicka 'Bygg/uppdatera index'.")
        return corpus

    for pdf in tqdm(pdfs, desc="Läser PDF:er"):
        raw = read_pdf_text(pdf)
        chunks = chunk_text(raw, CHUNK_SIZE, CHUNK_OVERLAP)
        for i, ch in enumerate(chunks, start=1):
            corpus.append({
                "text": ch,
                "source": pdf.name,
                "chunk_id": f"{pdf.name}#chunk{i}",
            })
    return corpus


# Cachea embedding-modellen (hämtas första gången från HF)

@st.cache_resource(show_spinner=True)
def get_embedder():
    """
    Laddar sentence-transformers-modellen en gång och cachar den i Streamlit.
    Modellen hämtas från Hugging Face (behöver internet första gången).
    """
    return SentenceTransformer(EMBED_MODEL_NAME)

# Bygg FAISS-index av textbitarna

def build_faiss(corpus: List[Dict], embedder):
    """
    Skapar ett vektorindex (FAISS) ovanpå embeddings av våra textbitar.
    Vi använder cosine-likhet via normaliserade embeddings + inner product.
    """
    if not corpus:
        return None, None, None

    texts = [c["text"] for c in corpus]
    # encode() returnerar np.array shape (N, dim)
    embs = embedder.encode(texts, normalize_embeddings=True)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product (IP) funkar med normaliserade vektorer ~ cosine
    index.add(embs.astype(np.float32))
    return index, embs.astype(np.float32), texts


# Retrieval: hämta TOP_K relevanta chunkar för en fråga

def retrieve(query: str, index, embedder, corpus: List[Dict], k: int = TOP_K) -> List[Dict]:
    """
    Vektor-sökning: frågeembedding vs FAISS-index för att få topp-k träffar.
    Returnerar en lista av dictar med text, källa, chunk-id och score.
    """
    if index is None or not corpus:
        return []

    q = embedder.encode([query], normalize_embeddings=True).astype(np.float32)
    D, I = index.search(q, k)
    I = I[0].tolist()  # top-k index
    results = []
    for rank, idx in enumerate(I, start=1):
        if idx < 0 or idx >= len(corpus):
            continue
        item = corpus[idx]
        results.append({
            "rank": rank,
            "score": float(D[0][rank - 1]),
            "text": item["text"],
            "source": item["source"],
            "chunk_id": item["chunk_id"],
        })
    return results


# Bygg en strikt systemprompt som begränsar svaret till KONTEKSTEN

def build_system_prompt(context_blocks: List[str], strict: bool = True) -> str:
    """
    Påminner modellen om att bara använda given kontext. Svenskt, kort, sakligt.
    """
    rules = textwrap.dedent("""
    Du är en assistent för Vårdcentralen Solrosen.
    Du får ENDAST svara utifrån KONTEKSTEN nedan.
    Regler:
    - Svara kort och tydligt på svenska (3–6 meningar).
    - Använd endast fakta som finns i KONTEKSTEN.
    - Om KONTEKSTEN inte innehåller svaret: säg tydligt att uppgiften saknas,
      och föreslå kontakt med 1177 eller vårdcentralen.
    - Lista "Källor" med 1–3 relevanta stycken (filnamn#chunk).
    - Lägg till en kort påminnelse om att detta inte ersätter medicinsk bedömning.
    """).strip()

    context = "\n\n---\n\n".join([f"[{i+1}] {cb}" for i, cb in enumerate(context_blocks)])
    if strict:
        rules += "\n- Om en uppgift inte finns i KONTEKSTEN ska du uttryckligen säga det."
    return f"{rules}\n\nKONTEKST:\n{context}"


# LLM-anrop via Ollama (lokalt)

def answer_with_llm(user_query: str, context_snippets: List[Dict], model_name: str) -> str:
    """
    Anropar lokal modell via Ollama, förser den med vår systemprompt + användarfråga.
    Lägger till källor och varningstext i slutet.
    """
    blocks = [snip["text"] for snip in context_snippets]
    sys_prompt = build_system_prompt(blocks, strict=True)

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_query},
    ]

    try:
        resp = ollama.chat(
            model=model_name,
            messages=messages,
            options={
                "temperature": 0.2,  # låg temp för sakliga svar
                "num_ctx": 4096      # kontextstorlek (justera efter modell)
            },
        )
        content = resp["message"]["content"].strip()
    except Exception as e:
        st.error(
            "Kunde inte nå Ollama. Är Ollama installerat och igång?\n"
            "Testa i terminal: `ollama list` (visa modeller), `ollama run llama3:instruct` (snabbtest).\n\n"
            f"Teknisk info: {e}"
        )
        st.stop()

    # Lägg till källor och standard-disclaimer
    sources_list = [f"{snip['source']} ({snip['chunk_id']})" for snip in context_snippets[:3]]
    return content


# UI (Streamlit)


with st.sidebar:

    rebuild = st.button("Uppdatera index")

    st.divider()
    st.subheader("LLM-inställningar")
    model_name = st.text_input("Ollama-modell", value=os.getenv("OLLAMA_MODEL", OLLAMA_MODEL_DEFAULT))
    st.caption("Exempel: llama3:instruct, llama3.1:8b-instruct, phi3:instruct, qwen2:7b-instruct")

    st.divider()
    st.subheader("Kontakt")
    st.write("Telefon: 010-123 45 67\n\nRådgivning: 1177\n\nAkut: 112")

# -------- Ladda/Bygg index (vid behov) --------
if rebuild or "corpus" not in st.session_state:
    with st.spinner("Läser PDF:er och bygger index..."):
        corpus = build_corpus_from_pdfs()
        embedder = get_embedder()
        index, embs, texts = build_faiss(corpus, embedder)
        st.session_state["corpus"] = corpus
        st.session_state["faiss"] = (index, embedder)
    st.success("Index klart!")

corpus = st.session_state.get("corpus", [])
faiss_pack = st.session_state.get("faiss", (None, None))
index, embedder = faiss_pack

# -------- Chatfönster --------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Visa tidigare meddelanden
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Inmatning
user_q = st.chat_input("Ställ en fråga (t.ex. 'När har provtagningen öppet?')")
if user_q:
    st.session_state["messages"].append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    # Hämtning av kontext (RAG)
    with st.spinner("Hämtar relevanta stycken..."):
        hits = retrieve(user_q, index, embedder, corpus, k=TOP_K)

    # Visa vad vi faktiskt hittade (transparens)
    with st.expander("Visa matchande stycken (RAG-kontext)"):
        if not hits:
            st.write("Inga matchningar i indexet. Lägg till/ladda upp PDF:er och bygg index.")
        for h in hits:
            st.markdown(f"**{h['rank']}. {h['source']}** — {h['chunk_id']}  \n_Score: {h['score']:.3f}_")
            st.code(h["text"])

    # Generera svar strikt utifrån kontext
    with st.spinner("Genererar svar utifrån kontext..."):
        answer = answer_with_llm(user_q, hits, model_name=model_name)

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state["messages"].append({"role": "assistant", "content": answer})

# Liten footer
st.caption("© Vårdcentralen Solrosen – Demo. Denna app använder lokal LLM via Ollama och läser endast dina PDF-dokument som kontext.")
