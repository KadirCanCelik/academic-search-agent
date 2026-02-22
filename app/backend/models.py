from pydantic import BaseModel
from typing import List, Optional

class QueryInput(BaseModel):
    """
    Schema for the incoming user request
    """
    input: str
    chat_history: Optional[List[dict]] = []

class AgentResponse(BaseModel):
    """
    Schema for the API response sent back to the client
    """
    output: str
    intermediate_steps: Optional[List[str]] = []