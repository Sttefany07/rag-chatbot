from __future__ import annotations
import math
import requests
from typing import List, Dict, Any
from app.settings import settings

"""
Convierte textos en vectores numericos (embeddings) usando Ollama
"""

# Clase base, como una "interfaz" para los embeddings
class EmbeddingsProvider:
    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

# Clase que usa Ollama para generar embeddings
class OllamaEmbeddings(EmbeddingsProvider):
    def __init__(self, host: str, model: str, timeout: int = 180):
        # Guardamos los datos de conexión
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()

    def _parse_embedding_response(self, data: Dict[str, Any]) -> List[float] | None:
        # Esta función busca el embedding dentro de la respuesta de Ollama
        if isinstance(data, dict):
            if "embedding" in data and isinstance(data["embedding"], list):
                return data["embedding"]
            if "data" in data and isinstance(data["data"], list) and data["data"]:
                item = data["data"][0]
                if isinstance(item, dict) and isinstance(item.get("embedding"), list):
                    return item["embedding"]
            if "embeddings" in data and isinstance(data["embeddings"], list) and data["embeddings"]:
                first = data["embeddings"][0]
                if isinstance(first, list):
                    return first
        return None

    def _call_ollama(self, payload: Dict[str, Any]) -> List[float] | None:
        # Hacemos la llamada POST al servidor de Ollama
        r = self.session.post(f"{self.host}/api/embeddings", json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return self._parse_embedding_response(data)

    def embed(self, texts: List[str]) -> List[List[float]]:
        # Generamos embeddings para cada texto de la lista
        outs: List[List[float]] = []
        for t in texts:
            t_norm = (t or "").strip()   # quitamos espacios
            if not t_norm:
                outs.append([])   #lista vacia
                continue

            # se intenta 'prompt'
            vec = self._call_ollama({"model": self.model, "prompt": t_norm})

            # se reintento con 'input'
            if not vec:
                vec = self._call_ollama({"model": self.model, "input": t_norm})

            if not vec:
                # si sigue sin devolver nada, error
                raise RuntimeError(
                    f"Ollama no devolvió embedding válido (modelo='{self.model}', len(text)={len(t_norm)})."
                )
            outs.append(vec)

        # Validamos que no haya embeddings vacíos

        if not outs or any((not v) for v in outs):
            raise RuntimeError("Se generaron embeddings vacíos. Revisa el modelo y los textos de entrada.")

        # Validamos que todos tengan la misma dimensión

        dim = len(outs[0])
        if any(len(v) != dim for v in outs):
            raise RuntimeError("Embeddings con dimensiones inconsistentes.")
        return outs


def get_embeddings() -> EmbeddingsProvider:
    if settings.embeddings_provider.lower() == "ollama":
        return OllamaEmbeddings(settings.ollama_host, settings.ollama_embed_model)
    raise NotImplementedError(f"Embeddings provider '{settings.embeddings_provider}' no implementado")


"""
Similitud coseno sirve para medir qué tan parecidos son dos vectore
"""
# Función para calcular la similitud coseno entre dos vectores
def cosine_sim(a: List[float], b: List[float]) -> float:
    # hacemos el producto punto
    dot = sum(x * y for x, y in zip(a, b))
    # calculamos la magnitud de cada vector
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    # si alguno es 0 devolvemos 0, sino hacemos la división
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)