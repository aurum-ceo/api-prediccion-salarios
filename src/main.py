"""
Main entry point for the FastAPI application.
"""
import uvicorn
from fastapi import FastAPI
from src.api import routes

app = FastAPI(
    title="Salary Prediction API",
    description="API for predicting salaries in data science using machine learning.",
    version="0.1.0",
)

# Include API routes
app.include_router(routes.router, prefix="/api/v1")


@app.get("/", tags=["root"])
async def read_root() -> dict:
    """
    Root endpoint returning basic API information.
    """
    return {
        "message": "Welcome to the Salary Prediction API",
        "docs": "/docs",
        "redoc": "/redoc",
        "version": app.version,
    }


@app.get("/health", tags=["health"])
async def health_check() -> dict:
    """
    Health check endpoint for monitoring.
    """
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
