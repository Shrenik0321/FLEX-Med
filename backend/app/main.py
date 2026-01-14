from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import predict, clients, federated

app = FastAPI(title="FLEX-Med Backend", version="0.1.0")

# CORS - adjust origins as needed (currently allow all for dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def healthcheck():
    return {"status": "ok"}

app.include_router(predict.router, prefix="/api")
app.include_router(clients.router, prefix="/api")
app.include_router(federated.router, prefix="/api")

