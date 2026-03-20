from pydantic import BaseModel


class MoveRequest(BaseModel):
    uci: str