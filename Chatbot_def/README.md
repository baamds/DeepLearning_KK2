# Vårdcentralen Solrosen – RAG‑assistent (Lokal, Streamlit + Ollama)

En helt **lokal** chattbot som svarar **endast** utifrån dina egna **PDF‑dokument** via RAG (Retrieval‑Augmented Generation).  

## Krav

- **Python** 3.10+ (testat med 3.11)
- **Ollama** installerat lokalt (Windows/macOS/Linux)
- Internet **endast första gången** för att hämta embedding‑modellen (Hugging Face) och Ollama‑modellen

**Python‑paket** (se `requirements.txt`):
```
streamlit==1.37.1
faiss-cpu==1.8.0
sentence-transformers==3.0.1
pypdf==4.3.1
ollama>=0.2.0
numpy==1.26.4
python-dotenv==1.0.1
tqdm==4.66.5
```

---

## Installation steg‑för‑steg

### 1) Installera Ollama
**Windows (Winget):**
```powershell
winget install Ollama.Ollama
```

**macOS (Homebrew):**
```bash
brew install --cask ollama
```


## Köra appen
I projektmappen:
```bash
streamlit run app.py
```
Öppna länken som visas (oftast `http://localhost:8501`).



**Bygg om index:**
- Klicka **“Bygg/uppdatera index”** i sidomenyn varje gång du ändrat dokumenten.




### PDF med skannade sidor (ingen text extraheras)
- **Orsak**: Bild‑baserade PDF:er saknar textlager.
- **Lösning (OCR)**: Förbehandla med t.ex. Tesseract (Pythonpaket: `pytesseract`, `pdf2image`).  
  *Idé*: Lägg till en importfunktion som kör OCR och skapar en sökbar PDF/TXT.





