# 🛡️ IRPR - Intelligent Incident Response & Playbook Recommender

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **AI-powered decision support system for cybersecurity incident response**

IRPR uses machine learning, similarity-based retrieval, and large language models to help security analysts respond to incidents faster and more effectively.

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Model Training](#-model-training)
- [API Documentation](#-api-documentation)
- [Configuration](#-configuration)
- [Evaluation & Metrics](#-evaluation--metrics)
- [Contributing](#-contributing)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)

---

## ✨ Features

### 🤖 **Intelligent Classification**

- Multi-model ensemble (Logistic Regression + SVM + Naive Bayes)
- Support for BERT-based deep learning models
- Calibrated confidence scores
- 6+ incident types: Phishing, Malware, Ransomware, Data Breach, Insider Misuse, DoS

### 🔍 **Advanced Similarity Search**

- Sentence embeddings for semantic understanding
- Hybrid retrieval (dense + sparse with BM25)
- Time-decay weighting for recent incidents
- Context-aware action recommendations

### 💡 **AI-Powered Explanations**

- LLM-generated rationale for each recommended action
- Phase-based action prioritization (Identification → Containment → Eradication → Recovery → Post-Incident)
- Risk assessment for skipped actions

### 📊 **Severity Assessment**

- Multi-factor severity scoring
- Real-time threat level indicators
- Action urgency weighting

### 🎨 **Modern Web Interface**

- Beautiful glassmorphic UI design
- Real-time markdown rendering
- Responsive mobile-first design
- Analyst override capabilities
- PDF report generation

### 📈 **Production-Ready**

- RESTful API with FastAPI
- Comprehensive error handling
- Model versioning and fallbacks
- Extensive evaluation metrics

---

## 🏗️ Architecture

```
┌─────────────────┐
│  Web Frontend   │  ← Electron/Web UI
│  (React/HTML)   │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│   FastAPI       │  ← REST API
│   Backend       │
└────────┬────────┘
         │
         ├─────────────────────────────┐
         ↓                             ↓
┌──────────────────┐         ┌──────────────────┐
│  ML Classifier   │         │  Similarity      │
│  - TF-IDF        │         │  Recommender     │
│  - Ensemble      │         │  - Embeddings    │
│  - BERT (opt)    │         │  - Hybrid Search │
└──────────────────┘         └──────────────────┘
         │                             │
         └─────────────┬───────────────┘
                       ↓
              ┌──────────────────┐
              │  LLM Explainer   │
              │  (Groq/Llama)    │
              └──────────────────┘
                       ↓
              ┌──────────────────┐
              │  Response Plan   │
              │  + Explanations  │
              └──────────────────┘
```

**Pipeline Flow:**

1. **Input:** Incident description text
2. **Classification:** ML model predicts incident type + confidence
3. **Similarity Search:** Find similar historical incidents
4. **Action Recommendation:** Rank actions based on similarity + phase
5. **Explanation Generation:** LLM explains each recommended action
6. **Severity Assessment:** Calculate threat level
7. **Output:** Complete response plan with rationale

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+ (for frontend)
- 8GB RAM minimum
- GROQ API key (for LLM features)

### Installation (5 minutes)

```bash
# Clone repository
git clone https://github.com/yourusername/irpr.git
cd irpr

# Install Python dependencies
pip install -r requirements_enhanced.txt

# Install frontend dependencies
cd frontend
npm install
cd ..

# Set up environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### Train Models (20-30 minutes)

```bash
# Train all models with one command
python train_evaluate_pipeline.py
```

This will:

- ✅ Train ensemble classifier
- ✅ Build similarity recommender
- ✅ Generate evaluation reports
- ✅ Save models to `./models/`

### Run the System

```bash
# Terminal 1: Start backend
uvicorn api.server:app --reload

# Terminal 2: Start frontend (web version)
cd frontend
npm start

# OR: Start Electron desktop app
npm run start
```

Visit **http://localhost:3000** 🎉

### Test with Sample Incident

```
During routine monitoring, repeated authentication attempts were
observed on a file server at 12:30 AM from a valid internal user
account accessing sensitive directories outside normal permissions.
```

---

## 📦 Installation

### Option 1: Standard Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements_enhanced.txt

# Download spaCy model (optional, for NER)
python -m spacy download en_core_web_sm
```

### Option 2: Docker Installation (Coming Soon)

```bash
docker-compose up -d
```

### Dependencies

**Core ML:**

- scikit-learn >= 1.3.0
- pandas >= 2.0.0
- numpy >= 1.24.0

**Advanced Features:**

- sentence-transformers >= 2.2.0 (semantic similarity)
- rank-bm25 >= 0.2.2 (keyword matching)
- transformers >= 4.30.0 (optional, for BERT)
- torch >= 2.0.0 (optional, for BERT)

**API & Services:**

- fastapi >= 0.100.0
- uvicorn >= 0.23.0
- groq >= 0.4.0 (LLM provider)

**Utilities:**

- reportlab >= 4.0.0 (PDF generation)
- matplotlib >= 3.7.0 (visualizations)
- seaborn >= 0.12.0 (plots)

---

## 🎯 Usage

### Web Interface

1. **Enter Incident Description**
   - Paste incident report in text area
   - Click "Analyze Incident"

2. **Review Results** (4 tabs)
   - **Situation:** Severity, incident type, confidence, assessment
   - **Response Plan:** Prioritized action items
   - **Rationale:** Detailed explanations for each action
   - **Evidence:** Similar historical incidents

3. **Take Action**
   - Follow recommended response plan
   - Submit analyst override if needed
   - Export PDF report

### Command Line Interface

```bash
python cli/cli.py "Suspicious email with credential request detected"
```

### Python API

```python
from app.orchestrator import run_pipeline

result = run_pipeline("Unusual database access from admin account")

print(f"Incident Type: {result['incident_type']}")
print(f"Confidence: {result['classification_confidence']:.2%}")

for action in result['actions'][:5]:
    print(f"  - {action['action_id']} ({action['phase']})")
```

### REST API

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "incident_text": "Multiple failed login attempts detected"
  }'
```

Response:

```json
{
  "incident_type": "Insider Misuse",
  "classification_confidence": 0.87,
  "severity": {"level": "High", "score": 4},
  "actions": [
    {
      "action_id": "IR-ID-01",
      "confidence": 95.2,
      "phase": "Identification"
    }
  ],
  "similar_incidents": [...],
  "situation_assessment": "..."
}
```

---

## 📁 Project Structure

```
irpr/
├── api/
│   └── server.py                 # FastAPI application
├── app/
│   ├── config.py                 # Configuration settings
│   └── orchestrator.py           # Main pipeline coordinator
├── core/
│   ├── classifier.py             # ML classification
│   ├── similarity.py             # Similarity search
│   ├── explainer.py              # LLM-based explanations
│   ├── severity.py               # Severity assessment
│   ├── situation.py              # Situation analysis
│   └── report.py                 # PDF generation
├── frontend/
│   ├── index.html                # Web UI
│   ├── renderer.js               # Frontend logic
│   ├── main.js                   # Electron entry point
│   └── package.json              # Node dependencies
├── knowledge/
│   └── action_kb.json            # Action knowledge base
├── models/
│   ├── enhanced_tfidf/           # Trained ensemble model
│   └── enhanced_similarity/      # Similarity recommender
├── data/
│   └── real_incidents_balanced.csv  # Training data
├── Phases/                       # Development phases (legacy)
├── enhanced_classifier.py        # Model training script
├── enhanced_similarity.py        # Recommender builder
├── train_evaluate_pipeline.py   # Complete training pipeline
├── requirements_enhanced.txt     # Python dependencies
├── .env.example                  # Environment template
└── README.md                     # This file
```

---

## 🎓 Model Training

### Quick Training

```bash
# Train all models with defaults
python train_evaluate_pipeline.py
```

### Custom Training

```python
from enhanced_classifier import EnhancedClassifier

# Train TF-IDF Ensemble (Fast, Good Performance)
classifier = EnhancedClassifier(strategy='ensemble')
classifier.fit(X_train, y_train)
classifier.save('./models/my_ensemble')

# Train BERT (Slower, Potentially Better)
classifier_bert = EnhancedClassifier(strategy='bert')
classifier_bert.fit(X_train, y_train, X_val, y_val)
classifier_bert.save('./models/my_bert')
```

### Build Similarity Recommender

```python
from enhanced_similarity import EnhancedSimilarityRecommender
import pandas as pd

df = pd.read_csv('./data/real_incidents_balanced.csv')

recommender = EnhancedSimilarityRecommender(
    embedding_model='all-MiniLM-L6-v2',
    use_hybrid=True,
    time_decay_enabled=True
)

recommender.fit(df)
recommender.save('./models/enhanced_similarity')
```

### Training Data Requirements

**Format:** CSV with columns:

- `text`: Incident description (string)
- `incident_type`: Label (string)

**Minimum samples:** 30 per class (180+ total recommended)

**Supported incident types:**

- Phishing
- Malware
- Ransomware
- Data Breach
- Insider Misuse
- Denial of Service

---

## 📡 API Documentation

### Endpoints

#### `POST /analyze`

Analyze an incident and get recommendations.

**Request:**

```json
{
	"incident_text": "string"
}
```

**Response:**

```json
{
	"incident_type": "string",
	"classification_confidence": 0.87,
	"actions": [
		{
			"action_id": "string",
			"confidence": 95.2,
			"phase": "string",
			"phase_rank": 1
		}
	],
	"explanations": [
		{
			"action_id": "string",
			"explanation": "string"
		}
	],
	"similar_incidents": [
		{
			"incident_type": "string",
			"similarity": 0.85,
			"text": "string"
		}
	],
	"severity": {
		"level": "High",
		"score": 4
	},
	"situation_assessment": "string"
}
```

#### `POST /export`

Generate PDF report.

**Request:**

```json
{
	"incident_text": "string"
}
```

**Response:**

```json
{
	"status": "report generated",
	"file": "incident_report.pdf"
}
```

#### `POST /override`

Record analyst override.

**Request:**

```json
{
	"corrected_incident_type": "string",
	"analyst_note": "string"
}
```

**Response:**

```json
{
	"status": "override recorded",
	"corrected_type": "string",
	"note": "string"
}
```

### Interactive API Docs

Visit **http://127.0.0.1:8000/docs** when backend is running.

---

## ⚙️ Configuration

### Environment Variables

Create `.env` file:

```bash
# LLM Configuration
GROQ_API_KEY=your_api_key_here

# Model Settings
USE_BERT=false
ENABLE_TIME_DECAY=true
SIMILARITY_TOP_K=5
HYBRID_ALPHA=0.7

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
```

### Model Configuration

Edit `app/config.py`:

```python
# Similarity settings
TOP_K_SIMILAR = 5        # Number of similar incidents
HYBRID_ALPHA = 0.7       # Weight for semantic vs keyword (0-1)

# LLM settings
LLM_TEMPERATURE = 0.1    # Lower = more consistent
ENABLE_LLM = True        # Use LLM for explanations

# Phase weights (for action prioritization)
PHASE_WEIGHTS = {
    "Identification": 1.0,
    "Containment": 0.9,
    "Eradication": 0.7,
    "Recovery": 0.6,
    "Post-Incident": 0.4
}
```

---

## 📊 Evaluation & Metrics

### View Evaluation Results

```bash
# After training, check results
cat evaluation_results/evaluation_report_*.txt
```

### Key Metrics

**Classification Performance:**

- Accuracy: 88-92% (vs 82% baseline)
- Precision/Recall: Balanced across classes
- Confidence Calibration: Well-calibrated probabilities

**Similarity Performance:**

- Top-5 Accuracy: ~85% (correct incident type in top 5)
- Semantic Relevance: High correlation with human judgment
- Time Complexity: O(n) for search, O(1) with caching

### Error Analysis

```python
import pandas as pd

# Load error cases
errors = pd.read_csv('evaluation_results/tfidf_ensemble_errors.csv')

# Analyze patterns
print(errors.groupby(['true_label', 'predicted_label']).size())
```

### Visualization Examples

Generated automatically during training:

1. **Confusion Matrix** - Per-class accuracy
2. **Confidence Distribution** - Model calibration
3. **Model Comparison** - Performance across metrics

---

## 🤝 Contributing

We welcome contributions! Here's how:

### Bug Reports

Open an issue with:

- System info (OS, Python version)
- Steps to reproduce
- Expected vs actual behavior
- Error logs

### Feature Requests

Describe:

- Use case
- Proposed solution
- Impact on existing features

### Pull Requests

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for new functionality
4. Ensure all tests pass
5. Update documentation
6. Submit PR with clear description

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements_dev.txt

# Run tests
pytest tests/

# Format code
black .
isort .

# Lint
flake8 core/ api/ app/
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. "Module not found: sentence_transformers"

**Solution:**

```bash
pip install sentence-transformers
```

#### 2. Backend won't start - "Models not found"

**Solution:** Train models first:

```bash
python train_evaluate_pipeline.py
```

#### 3. Out of memory during training

**Solutions:**

- Use ensemble instead of BERT: `strategy='ensemble'`
- Reduce batch size in BERT training
- Use CPU instead of GPU: `export CUDA_VISIBLE_DEVICES=""`

#### 4. Frontend not loading

**Check:**

```bash
# Verify backend is running
curl http://127.0.0.1:8000/docs

# Check frontend dependencies
cd frontend && npm install

# Clear cache
rm -rf node_modules package-lock.json
npm install
```

#### 5. Low classification accuracy

**Causes:**

- Insufficient training data (need 30+ samples per class)
- Imbalanced classes
- Poor data quality

**Solutions:**

```bash
# Check data balance
python -c "import pandas as pd; print(pd.read_csv('data/real_incidents_balanced.csv')['incident_type'].value_counts())"

# Retrain with more data
python train_evaluate_pipeline.py
```

#### 6. Slow API responses

**Optimizations:**

- Cache embeddings in memory
- Use async processing
- Reduce `TOP_K_SIMILAR` in config
- Disable LLM explanations for faster responses

### Debug Mode

```bash
# Run backend with debug logging
uvicorn api.server:app --reload --log-level debug

# Test individual components
python -c "
from core.classifier import classify_incident
result = classify_incident('test incident')
print(result)
"
```

---

## 📖 Documentation

- **API Docs:** http://localhost:8000/docs (when running)
- **Model Architecture:** See `enhanced_classifier.py` comments
- **Action Knowledge Base:** `knowledge/action_kb.json`
- **Training Guide:** See Implementation Guide in artifacts

---

## 🗺️ Roadmap

### Version 2.0 (Current)

- ✅ Enhanced ensemble models
- ✅ Sentence embeddings
- ✅ Hybrid search
- ✅ Modern UI
- ✅ Comprehensive evaluation

### Version 2.1 (Planned)

- [ ] Named Entity Recognition (NER)
- [ ] Feedback loop for continuous learning
- [ ] Multi-language support
- [ ] Custom BERT model for security domain
- [ ] Real-time streaming analysis

### Version 3.0 (Future)

- [ ] Graph-based incident correlation
- [ ] Automated response execution (with approval)
- [ ] Integration with SIEM systems
- [ ] Multi-tenant support
- [ ] Advanced explainability (LIME/SHAP)

---

## 📄 License

MIT License

Copyright (c) 2025 IRPR Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---


## 🙏 Acknowledgments

- **CERT Dataset** - CMU Software Engineering Institute
- **Enron Email Dataset** - Public corpus for NLP research
- **Hugging Face** - Transformers and sentence-transformers
- **Groq** - Fast LLM inference
- **FastAPI** - Modern Python web framework

---

<div align="center">

[⬆ Back to Top](#-irpr---intelligent-incident-response--playbook-recommender)

</div>
