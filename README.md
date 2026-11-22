# voiceflow-ai

Two FastAPI services for call-center audio: speech-to-text with Whisper, and classification of the transcript into call outcome labels, tuned for insurance calls (Medicare, ACA, Final Expense).

## Features

- Transcription service (`POST /transcribe/`): receives an audio file, transcribes it with faster-whisper (`small.en`, GPU float16 or CPU int8), then labels the text. Uses context prompts that depend on call type and turn number.
- Labelling pipeline: cleans the transcript (contractions, noise phrases), tries exact and substring phrase lists first, and only then calls the classification service.
- Classification service (`POST /classify/`): runs fine-tuned DistilBERT sequence classifiers, chosen by `call_type` (`medicare`, `aca`, `fe`) and `model_type`.
- `GET /health` on both services and `POST /shutdown` for a graceful drain of running requests.
- JSON logging with rotating log files, and Docker images plus a Docker Swarm stack file.

## Tech stack

Python 3.11, FastAPI, Uvicorn, faster-whisper, transformers and PyTorch (CUDA 11.8 images), Docker / Docker Swarm.

## Project structure

```
voiceflow_ai/
  transcription_app.py, classification_app.py   the two FastAPI apps
  routers/        /transcribe/ and /classify/ endpoints
  services/       Whisper and DistilBERT wrappers
  core/           config (model paths, phrase lists), logger, transcript processing
Dockerfile_transcription, Dockerfile_classification
docker-stack.yml
requirements.txt
```

## Setup and run

You must supply the trained models, which are not in this repository:

1. In `voiceflow_ai/core/config.py`, set the model locations (empty strings in the repo): `DISTIL_MODEL`, `MEDICARE_MODEL_B`, `MEDICARE_MODEL_11`, `MEDICARE_MODEL_12`, `ACA_MODEL`, `ACA_MODEL_B`, `FE_MODEL_B` (paths or Hugging Face model IDs for DistilBERT classifiers).
2. Put a short test recording at `voiceflow_ai/core/whisper_test.wav`. The transcription service runs it at start-up for its health check.

Run locally:

```bash
pip install -r requirements.txt
uvicorn voiceflow_ai.classification_app:app --port 9000
uvicorn voiceflow_ai.transcription_app:app --port 8000   # in a second terminal
```

The transcription processor calls the classifier at `http://voiceflow_classification:9000`, so for local runs add that hostname to `/etc/hosts` pointing at 127.0.0.1, or change the URL in `core/transcription_processor.py`.

Docker Swarm (NVIDIA GPU host):

```bash
docker build -f Dockerfile_transcription -t api-transcription:latest .
docker build -f Dockerfile_classification -t api-classification:latest .
docker swarm init
docker network create --driver overlay voiceflow-net
docker stack deploy -c docker-stack.yml voiceflow
```

Environment variables: the code reads none. `docker-stack.yml` sets `NVIDIA_VISIBLE_DEVICES=all`. No Google credentials are used by the code. If you add Google Cloud features, load the service account from the standard `GOOGLE_APPLICATION_CREDENTIALS` path (a file outside the repo). Key files are git-ignored.

## Usage example

```bash
curl -X POST http://localhost:8000/transcribe/ \
  -F uuid=abc-123 -F connection_id=call-1 -F turn_number=1 \
  -F model_type=A -F call_type=medicare -F file=@call.wav

curl -X POST http://localhost:9000/classify/ \
  -F transcribed_text="yes I have medicare" -F serial_number=call-1 \
  -F model_type=A -F call_type=medicare
```

The transcription response contains `transcription`, `label`, `confidence`, timings and `model_used`.

## Limitations and status

- Not runnable out of the box: models, the test audio and local-name settings must be provided (see Setup). Not tested here, since it needs the models and a GPU for realistic use.
- `docker-stack.yml` runs 9 transcription replicas and expects the service name `voiceflow_classification`; adjust for your hardware.
- The label codes (for example `N`, `AP`, `HP`) are specific to the original project and are not documented in the code.
- A Google service account key that was committed has been removed from the working tree. It remains in git history, so treat it as compromised and revoke it.

## License

MIT, see [LICENSE](LICENSE).
