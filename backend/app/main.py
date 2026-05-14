from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
from app.routers import detection
from app.schemas.detection import HealthResponse
from app.utils.logger import get_logger

#Load settings and logger
settings = get_settings()
logger = get_logger(__name__)

def create_app() -> FastAPI:
    """
    Application factory function
    Creates and configures the FastAPI instance
    """

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="AI-powered fake news and miinformation detector",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    #CORS Middleware 
    #Allows the React frontend to call this backend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.frontend_url],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    #Register Routers 
    app.include_router(detection.router)

    #Startup Event
    @app.on_event("startup")
    async def startup_event():
        logger.info(f"🚀 {settings.app_name} v{settings.app_version} starting up...")
        logger.info(f"📖 API docs available at: http://localhost:8000/docs")


    #Shutdown Event
    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Truthlens shutting down")

    return app

#Create the app instance 
app = create_app()

#Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Simple health check to verify the server is running
    Used by deployment platforms to monitor the service
    """
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        app_name=settings.app_name
    )

                  