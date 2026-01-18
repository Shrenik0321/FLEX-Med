"""FLEX-Med FastAPI Application Entry Point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import clients, federated, predict
from app.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="Federated Learning for Medical Imaging - Backend API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(clients.router, prefix="/api", tags=["Clients"])
app.include_router(federated.router, prefix="/api", tags=["Federated Learning"])
app.include_router(predict.router, prefix="/api", tags=["Predictions"])


@app.get("/")
async def root():
    """Root endpoint returning API information."""
    return {
        "name": settings.app_name,
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
