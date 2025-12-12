from pydantic import BaseModel
from typing import Optional

class ClientBase(BaseModel):
    client_name: str
    client_email: str
    status: str
    model_type: str

class ClientCreate(ClientBase):
    pass

class Client(ClientBase):
    id: int

    class Config:
        orm_mode = True
