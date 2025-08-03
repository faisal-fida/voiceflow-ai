from voiceflow_ai.services.transcription_service import TranscriptionService
from voiceflow_ai.services.classification_service import ClassificationService

transcription_service = TranscriptionService()
classification_service = ClassificationService()


def get_transcription_service():
    return transcription_service


def get_classification_service():
    return classification_service
