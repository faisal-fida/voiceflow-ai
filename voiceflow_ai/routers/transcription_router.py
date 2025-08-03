import asyncio
import os
import time
from tempfile import NamedTemporaryFile

import aiofiles
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form

from voiceflow_ai.core.logger import get_logger
from voiceflow_ai.core.dependencies import get_transcription_service
from voiceflow_ai.core.transcription_processor import TranscriptionProcessor
from voiceflow_ai.services.transcription_service import TranscriptionService

router = APIRouter()

transcription_processor = TranscriptionProcessor()

logger = get_logger("transcription_router")


@router.post("/transcribe/")
async def transcribe(
    transcription_service: TranscriptionService = Depends(get_transcription_service),
    uuid: str = Form(...),
    connection_id: str = Form(...),
    turn_number: str = Form(...),
    model_type: str = Form(...),
    call_type: str = Form(...),
    file: UploadFile = File(...),
):
    if turn_number is not None:
        turn_number = int(turn_number)
    else:
        turn_number = 1

    if model_type is None:
        model_type = "A"

    if transcription_service.shutdown_in_progress:
        logger.debug(
            "Shutdown request received, rejecting the request",
            extra={"serial_number": connection_id},
        )
        raise HTTPException(
            status_code=503,
            detail="Server is shutting down. No new requests are being accepted.",
        )
    temp_file = None
    transcribed_text = None
    processed_transcribed_text = None
    try:
        model_used = "None"
        # Save uploaded file to a temporary file
        temp_file = NamedTemporaryFile(delete=False)
        async with aiofiles.open(temp_file.name, mode="wb") as f:
            await f.write(await file.read())
        temp_file.close()
        audio_file_path = temp_file.name
        print(f"Temporary file created at {temp_file.name}")

        start_time = time.time()
        loop = asyncio.get_running_loop()
        try:
            transcribed_text = await loop.run_in_executor(
                None,
                transcription_service.transcribe_audio,
                audio_file_path,
                call_type,
                model_type,
                turn_number,
            )
        except Exception as e:
            logger.error(
                f"An error occurred during transcription in endpoint: {e}",
                extra={"serial_number": connection_id},
            )
        end_time = time.time()
        transcription_time = end_time - start_time
        logger.debug(
            f"Transcription completed: {transcribed_text} and the time is: {transcription_time}",
            extra={"serial_number": connection_id},
        )

        classification_start_time = time.time()
        e = None
        try:
            (
                label,
                confidence,
                processed_transcribed_text,
                model_used,
            ) = await loop.run_in_executor(
                None,
                transcription_processor.process_transcription,
                transcribed_text,
                connection_id,
                model_type,
                call_type,
                turn_number,
            )
        except Exception as e:
            logger.error(
                f"An error occurred during classification in endpoint: {e}",
                extra={"serial_number": connection_id},
            )
            label = "N"
            confidence = 1.5

        classification_time = time.time() - classification_start_time

        response_data = {
            "uuid": uuid,
            "transcription": transcribed_text,
            "label": label,
            "confidence": confidence,
            "transcription_time": transcription_time,
            "classification_time": classification_time,
            "processed_transcribed_text": processed_transcribed_text,
            "model_used": model_used,
            "error": e,
        }

        return response_data
    except Exception as e:
        logger.error(
            f"An error occurred during transcription: {e}",
            extra={"serial_number": connection_id},
        )
        if temp_file is not None:
            # Delete the temporary file
            os.unlink(temp_file.name)
        raise HTTPException(status_code=500, detail=str(e))
