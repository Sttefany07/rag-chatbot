from __future__ import annotations
import requests
from typing import List, Dict, Any
from app.settings import settings

"""
La app se conecta con Ollama, manda las preguntas junto al contexto, 
y devuelve la respuesta generada por el modelo
"""

class LLMClient:
    def __init__(self, host: str, model: str, timeout: int = 300):  # 300s
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()

    # Función para chatear con el modelo
    def chat(self, messages: List[Dict[str, str]], stream: bool = False) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": 0.2,
                "num_ctx": 8192,   # prueba 8192; si tu hw soporta más, sube a 16384
            },
        }
        # mandamos la request a Ollama
        r = self.session.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()

        if isinstance(data, dict):
            if "message" in data and isinstance(data["message"], dict):
                return {"content": data["message"].get("content", ""), "model": self.model}
            if "content" in data:
                return {"content": data.get("content", ""), "model": self.model}
        return {"content": str(data), "model": self.model}

def get_llm() -> LLMClient:
    if settings.llm_provider.lower() == "ollama":
        return LLMClient(settings.ollama_host, settings.ollama_model)
    raise NotImplementedError(f"LLM provider '{settings.llm_provider}' no implementado")
