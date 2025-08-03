# VoiceFlow AI

🎙️ **AI-Powered Voice Analytics Platform for Insurance Industry**

VoiceFlow AI is a production-ready FastAPI microservices platform that provides real-time speech-to-text transcription and intelligent text classification, specifically optimized for insurance call center analytics.

## 🌟 Key Features

### 🎯 **Dual-Service Architecture**
- **Transcription Service**: High-performance speech-to-text using OpenAI Whisper
- **Classification Service**: Intent/outcome classification using specialized DistilBERT models

### 🏥 **Industry-Specialized Models**
- **Medicare**: Optimized for Medicare-related calls and terminology
- **ACA (Affordable Care Act)**: Healthcare marketplace conversations  
- **Final Expense**: Life insurance and burial coverage discussions

### 🧠 **Context-Aware Processing**
- Dynamic prompts based on call type and conversation turn
- Specialized models for different conversation stages
- Industry-specific vocabulary and terminology handling

### ⚡ **Production-Ready Features**
- GPU acceleration with automatic CPU fallback
- Docker Swarm deployment with health checks
- Graceful shutdown and resource management
- Comprehensive logging and monitoring
- Async processing for high throughput

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- NVIDIA GPU (optional, will fallback to CPU)
- Python 3.8+ (for development)

### 🐳 Docker Deployment

1. **Clone the repository**
```bash
git clone <repository-url>
cd voiceflow-ai
```

2. **Build the services**
```bash
# Build transcription service
docker build -f Dockerfile_transcription -t api-transcription:latest .

# Build classification service  
docker build -f Dockerfile_classification -t api-classification:latest .
```

3. **Deploy with Docker Swarm**
```bash
# Initialize swarm (if not already done)
docker swarm init

# Create external network
docker network create --driver overlay voiceflow-net

# Deploy the stack
docker stack deploy -c docker-stack.yml voiceflow
```

### 🔧 Development Setup

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Configure models** (Update `voiceflow_ai/core/config.py`)
```python
# Set your model paths
DISTIL_MODEL = "path/to/your/classification/model"
MEDICARE_MODEL_B = "path/to/medicare/model"
ACA_MODEL = "path/to/aca/model"
# ... other model configurations
```

3. **Run services locally**
```bash
# Transcription service (port 8000)
uvicorn voiceflow_ai.transcription_app:app --host 0.0.0.0 --port 8000

# Classification service (port 9000)
uvicorn voiceflow_ai.classification_app:app --host 0.0.0.0 --port 9000
```

## 📡 API Endpoints

### Transcription Service (Port 8000)

#### `POST /transcribe/`
Convert audio to text with intelligent classification

**Request:**
```bash
curl -X POST "http://localhost:8000/transcribe/" \
  -F "file=@audio.wav" \
  -F "uuid=unique-call-id" \
  -F "connection_id=conn-123" \
  -F "turn_number=1" \
  -F "model_type=A" \
  -F "call_type=medicare"
```

**Response:**
```json
{
  "uuid": "unique-call-id",
  "transcription": "Hello, I'm calling about Medicare benefits",
  "label": "P",
  "confidence": 0.89,
  "transcription_time": 2.1,
  "classification_time": 0.3,
  "processed_transcribed_text": "hello i'm calling about medicare benefits",
  "model_used": "mc_10.3"
}
```

### Classification Service (Port 9000)

#### `POST /classify/`
Classify transcribed text for call analysis

**Request:**
```bash
curl -X POST "http://localhost:9000/classify/" \
  -F "transcribed_text=I'm interested in Medicare plans" \
  -F "serial_number=conn-123" \
  -F "model_type=A" \
  -F "call_type=medicare"
```

**Response:**
```json
{
  "label": "P",
  "confidence": 0.92,
  "model_used": "mc_10.3"
}
```

### Health Checks
- `GET /health` - Service health status
- `POST /shutdown` - Graceful shutdown

## 🏗️ Architecture

```
┌─────────────────────┐    ┌─────────────────────┐
│   Client/Frontend   │    │   Load Balancer     │
└──────────┬──────────┘    └──────────┬──────────┘
           │                          │
           └──────────────────────────┘
                      │
         ┌────────────┴────────────┐
         │                         │
┌────────▼────────┐    ┌───────────▼──────────┐
│ Transcription   │    │  Classification      │
│ Service         │    │  Service             │
│ (Port 8000)     │    │  (Port 9000)         │
│                 │    │                      │
│ ┌─────────────┐ │    │ ┌──────────────────┐ │
│ │   Whisper   │ │    │ │   DistilBERT     │ │
│ │   Model     │ │    │ │   Models         │ │
│ └─────────────┘ │    │ │  - Medicare      │ │
└─────────────────┘    │ │  - ACA           │ │
                       │ │  - Final Expense │ │
                       │ └──────────────────┘ │  
                       └──────────────────────┘
```

## 🔧 Configuration

Key configuration options in `voiceflow_ai/core/config.py`:

```python
class Settings:
    # Model paths
    DISTIL_MODEL = "path/to/base/classification/model"
    MEDICARE_MODEL_B = "path/to/medicare/model"
    ACA_MODEL = "path/to/aca/model"
    FE_MODEL_B = "path/to/final-expense/model"
    
    # Whisper configuration
    WHISPER_MODEL = "openai/whisper-tiny.en"
    
    # Processing type
    TYPE = True  # True for multi-class, False for 3-class classification
```

## 📊 Call Classification Labels

### Medicare Calls
- **P**: Positive/Interested
- **N**: Negative/Not Interested  
- **U**: Unclear/Uncertain
- **DNC**: Do Not Call
- **CB**: Call Back
- **AP**: Age-related responses
- And 70+ specialized labels...

### ACA Calls
- **ELI**: Eligible
- **SUB**: Subsidy-related
- **ICE**: Insurance coverage existing
- **ACA**: ACA-specific responses
- And 45+ specialized labels...

### Final Expense Calls
- **BENE**: Beneficiary discussions
- **COST**: Cost-related inquiries
- **SCAM**: Scam detection
- And 45+ specialized labels...

## 🚀 Performance & Scaling

### Resource Requirements
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: 8GB+ (16GB+ for production)
- **CPU**: 4+ cores
- **Storage**: 10GB+ for models

### Scaling Configuration
```yaml
# docker-stack.yml
services:
  transcription:
    deploy:
      replicas: 9  # Scale based on load
  classification:
    deploy:
      replicas: 1  # Lighter classification service
```

## 🔒 Security & Compliance

- Google Service Account integration for cloud services
- Secure file handling with automatic cleanup
- Request tracking with unique connection IDs
- Health monitoring and graceful shutdowns

## 📈 Monitoring & Logging

Comprehensive logging with:
- Request/response tracking
- Performance metrics (processing times)
- Error handling and debugging
- Serial number tracking for call tracing

## 🛠️ Development

### Project Structure
```
voiceflow_ai/
├── core/                 # Core utilities and config
│   ├── config.py        # Configuration settings
│   ├── dependencies.py  # Dependency injection  
│   ├── logger.py        # Logging configuration
│   └── transcription_processor.py
├── routers/             # API route handlers
│   ├── transcription_router.py
│   └── classification_router.py  
├── services/            # Business logic services
│   ├── transcription_service.py
│   └── classification_service.py
├── classification_app.py # Classification FastAPI app
└── transcription_app.py  # Transcription FastAPI app
```

### Adding New Models

1. **Update configuration** in `config.py`
```python
NEW_MODEL_PATH = "path/to/new/model"
```

2. **Initialize in service** 
```python
self.new_model = AutoModelForSequenceClassification.from_pretrained(NEW_MODEL_PATH)
```

3. **Add classification logic**
```python
if call_type == "new_type":
    model = self.new_model
    tokenizer = self.new_tokenizer
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI Whisper** for speech recognition capabilities
- **Hugging Face Transformers** for BERT-based classification
- **FastAPI** for the high-performance web framework
- **Docker** for containerization and deployment

## 🆘 Support

For support, email [your-email] or create an issue in the repository.

---

**VoiceFlow AI** - Transforming voice conversations into actionable insights for the insurance industry.
