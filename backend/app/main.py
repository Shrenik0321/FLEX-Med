"""FLEX-Med FastAPI Application Entry Point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import clients, fl_simulation, inference, client_simulation_metrics
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
app.include_router(fl_simulation.router, prefix="/api", tags=["FL Simulation"])
app.include_router(inference.router, prefix="/api", tags=["Inference"])
app.include_router(client_simulation_metrics.router)  # Has its own prefix
    
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
