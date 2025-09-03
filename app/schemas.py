from typing import List, Optional, Literal, Any
from pydantic import BaseModel

"""
define cómo deben verse los datos que entran y salen de la API.
validar requests del usuario y estructurar respuestas del asistente con sus fuentes
"""
# Un mensaje dentro de la conversación
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

# Lo que manda el usuario cuando hace una pregunta al chat
class ChatRequest(BaseModel):
    message: str
    top_k: int = 6
    mmr: Optional[bool] = None
    history: Optional[List[ChatMessage]] = None
    source: Optional[str] = None

# Un fragmento de documento recuperado como fuente
class SourceChunk(BaseModel):
    id: str
    source: str
    page: int | None = None
    distance: float | None = None
    text: str

# Lo que devuelve el chat como respuesta
class ChatResponse(BaseModel):
    answer: str
    used_sources: List[SourceChunk]
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    model: str | None = None
    extra: dict[str, Any] | None = None