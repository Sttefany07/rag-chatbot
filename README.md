# rag-chatbot
RAG Chatbot (FastAPI + Chroma + Ollama)

Es un proyecto donde puedes subir documentos (PDF, TXT, MD) y luego hacer preguntas en español.
El sistema busca la respuesta dentro de esos documentos usando inteligencia artificial (IA).

⚙️ Stack principal

  Python 3.10+ → Lenguaje principal del proyecto.
  FastAPI → Framework para la API (los endpoints como /ask).
  Ollama → Donde corres el LLM (modelo de lenguaje) y los embeddings.
      Modlo de LLM: llama3.1:8b
      Modelo de embeddings: nomic-embed-text
  ChromaDB → Base vectorial que guarda los embeddings de los documentos.
  pypdf → Para leer y extraer texto de archivos PDF.
  langchain_text_splitters → Para partir el texto en chunks (trozos manejables).
  Pydantic → Para definir y validar los schemas de entrada y salida en la API.
  dotenv → Para manejar configuraciones en el archivo .env.

🛠️ Requisitos
  Tener Python 3.10 o más.

Instalar Ollama
  y que esté corriendo en tu PC (ollama serve).

Descargar los modelos en Ollama:
  ollama pull llama3.1:8b
  ollama pull nomic-embed-text:latest


Instalar dependencias:

pip install -r requirements.txt

📂 Carpetas importantes
    app/
     ├─ embeddings.py   → convierte texto en vectores
     ├─ retriever.py    → busca en la base vectorial
     ├─ rag.py          → arma contexto para el LLM
     ├─ llm.py          → conecta con Ollama
     ├─ schemas.py      → define cómo son las requests/responses
     ├─ ingest.py       → lee PDFs y los guarda en Chroma
     ├─ settings.py     → configuraciones del proyecto
     └─ main.py         → API con FastAPI
    data/
     ├─ chroma/         → aquí se guarda la base de datos vectorial
     └─ uploads/        → aquí pones tus archivos (PDF, TXT, MD)

⚙️ Configuración
Crea un archivo .env en la raíz con esto:

  LLM_PROVIDER=ollama
  OLLAMA_HOST=http://localhost:11434
  OLLAMA_MODEL=llama3.1:8b
  EMBEDDINGS_PROVIDER=ollama
  OLLAMA_EMBED_MODEL=nomic-embed-text
  CHROMA_PATH=./data/chroma
  MAX_CONTEXT_CHUNKS=6
  MMR=true
  CHUNK_SIZE=1200
  CHUNK_OVERLAP=220
  INGEST_BATCH_SIZE=256


🚀 Cómo usarlo
  Levanta la API:
  uvicorn app.main:app --reload --port 8000
  Abre en tu navegador:
  http://localhost:8000/docs

🧩  Cómo se conecta todo
  ingest.py → lee PDF/TXT, los parte en chunks y los mete en Chroma como embeddings.
  retriever.py → busca en Chroma los chunks más parecidos a la pregunta.
  rag.py → arma el contexto con esos chunks y construye el prompt.
  llm.py → manda el prompt a Ollama y recibe la respuesta del modelo.
  schemas.py → define el formato de requests/responses.
  settings.py → lee la configuración desde .env.
  main.py → une todo y expone la API con FastAPI.
