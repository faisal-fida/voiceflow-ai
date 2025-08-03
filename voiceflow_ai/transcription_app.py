import asyncio
import requests
import time

from fastapi import FastAPI, BackgroundTasks, Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from voiceflow_ai.core.dependencies import get_transcription_service
from voiceflow_ai.routers import transcription_router
from voiceflow_ai.core.logger import get_logger
from voiceflow_ai.core.config import settings as c


app = FastAPI()

transcription_service = get_transcription_service()

logger = get_logger("transcription")

test = False


async def startup_event():
    global test
    test = transcription_service.initialize_model()
    logger.info("Transcription model initialized successfully")


async def shutdown_event():
    await transcription_service.shutdown()


app.add_event_handler("startup", startup_event)
app.add_event_handler("shutdown", shutdown_event)

app.include_router(transcription_router.router)


async def run_test_transcription():
    logger.info("Transcription test starting")
    await asyncio.sleep(1)
    logger.info("Transcription test starting x2")

    # Assume the test audio file is in the same directory as this script
    audio_file_path = c.MODEL_LOAD_FILEPATH

    # Open the file in binary mode
    with open(audio_file_path, 'rb') as f:
        audio_file = f.read()

    # Create the form data
    data = {
        "uuid": "test_uuid",
        "connection_id": "test",
        "turn_number": 1,
    }

    # Create the files dictionary
    files = {
        "file": ("whisper_test.wav", audio_file),
    }

    # Send a POST request to the /transcribe/ endpoint
    response = requests.post("http://127.0.0.1:8000/transcribe/", data=data, files=files)

    # Log the response
    logger.info(f"Test transcription response: {response.json()}")

    label = response.json().get('label')
    if label == 'N':
        test = True
    else:
        test = False

    return test


@app.get("/health")
async def health_check():
    global test
    if (transcription_service.whisper_model is not None and test and
        not transcription_service.shutdown_in_progress and
        (not transcription_service.active_transcriptions or
         time.time() - transcription_service.active_transcriptions[0] <= 8)):
        return {"status": "healthy"}
    else:
        raise HTTPException(status_code=500, detail="Service is unhealthy")


@app.post("/shutdown")
async def shutdown(background_tasks: BackgroundTasks):
    logger.info("Shutdown signal received")
    transcription_service.shutdown_in_progress = True
    background_tasks.add_task(shutdown_application)


async def shutdown_application():
    logger.debug(f"Shutdown signal received. Current transcriptions are: "
                 f"{transcription_service.active_transcriptions_count}")
    while transcription_service.active_transcriptions_count > 0:
        await asyncio.sleep(0.1)
    await transcription_service.shutdown()


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
