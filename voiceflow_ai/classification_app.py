import asyncio
import time
import sys

from fastapi import FastAPI, BackgroundTasks, Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from voiceflow_ai.routers import classification_router
from voiceflow_ai.core.dependencies import get_classification_service
from voiceflow_ai.core.logger import get_logger

app = FastAPI()

classification_service = get_classification_service()
logger = get_logger("classification")


async def startup_event():
    classification_service.initialize_model()
    logger.info("Classification model initialized successfully")


def shutdown_event():
    classification_service.shutdown()


app.add_event_handler("startup", startup_event)
app.add_event_handler("shutdown", shutdown_event)

app.include_router(classification_router.router)


@app.get("/health")
async def health_check():
    if (classification_service.distil_model is not None and
        not classification_service.shutdown_in_progress and
        (not classification_service.active_classifications or
         time.time() - classification_service.active_classifications[0] <= 1)):
        return {"status": "healthy"}
    else:
        raise HTTPException(status_code=500, detail="Service is unhealthy")


@app.post("/shutdown")
async def shutdown(background_tasks: BackgroundTasks):
    logger.info("Shutdown signal received")
    classification_service.shutdown_in_progress = True
    background_tasks.add_task(shutdown_application)


async def shutdown_application():
    logger.debug(f"Shutdown signal received. Current classifications are: "
                 f"{classification_service.active_classifications_count}")
    while classification_service.active_classifications_count > 0:
        await asyncio.sleep(0.1)
    await classification_service.shutdown()
    sys.exit(0)


@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    logger.error(f"An error occurred, general exception: Request: {request} and Exception: {exc}",
                 extra={"serial_number": request.get("serial_number")})
    return {"error": str(exc)}


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.url.path != "/health":
            logger.debug(f"Processing request: {request.method} {request.url}",
                         extra={"serial_number": request.get("serial_number")})
        response = await call_next(request)
        if request.url.path != "/health":
            logger.debug(f"Request processed: {response.status_code}",
                         extra={"serial_number": request.get("serial_number")})
        return response


app.add_middleware(LoggingMiddleware)
