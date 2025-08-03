import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException, Depends, Form

from voiceflow_ai.core.logger import get_logger
from voiceflow_ai.core.dependencies import get_classification_service
from voiceflow_ai.services.classification_service import ClassificationService

router = APIRouter()

logger = get_logger("classification_router")

# Create a ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=1)


@router.post("/classify/")
async def classify(classification_service: ClassificationService = Depends(get_classification_service),
                   transcribed_text: str = Form(...), serial_number: str = Form(...), model_type: str = Form(...),
                   call_type: str = Form(...)):
    connection_id = serial_number
    if classification_service.shutdown_in_progress:
        logger.debug("Shutdown request received, rejecting the request",
                     extra={"serial_number": connection_id})
        raise HTTPException(status_code=503, detail="Server is shutting down. No new requests are being accepted.")

    try:
        start_time = time.time()
        loop = asyncio.get_running_loop()
        label, confidence, model_used = await loop.run_in_executor(None, classification_service.classify_audio,
                                                                   transcribed_text, connection_id, model_type,
                                                                   call_type)
        end_time = time.time()
        classification_time = end_time - start_time

        response_data = {
            'label': label,
            'confidence': confidence,
            'model_used': model_used,
        }

        logger.debug(f"Label is: {label} and confidence is: {confidence} and time taken is: {classification_time}"
                     f" and model used is : {model_used}",
                     extra={"serial_number": connection_id})
        return response_data
    except Exception as e:
        logger.error(f"An error occurred during transcription: {e}",
                     extra={"serial_number": connection_id})
        raise HTTPException(status_code=500, detail=str(e))
