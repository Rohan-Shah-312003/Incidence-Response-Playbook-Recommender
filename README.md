# IRPR – Incident Response Decision Engine

## Overview

IRPR (Incident Response and Prevention) is a comprehensive system designed to assist security analysts in responding to incidents. It leverages a combination of machine learning, similarity-based reasoning, and retrieval-augmented generation (RAG) to provide a holistic incident response assistant.

The system takes a textual description of a security incident and provides:
-   **Classification:** Identifies the type of incident.
-   **Recommended Actions:** Suggests a course of action.
-   **Severity Assessment:** Estimates the incident's severity.
-   **Situation Assessment:** Provides a summary of the situation.
-   **Action Explanations:** Explains the reasoning behind the recommended actions.

## Features

-   **Incident Classification:** Automatically classifies incidents into predefined categories.
-   **Action Recommendation:** Recommends response actions based on the incident type and similar past incidents.
-   **Severity Scoring:** Calculates a severity score to help prioritize incidents.
-   **Natural Language Explanations:** Uses LLMs to provide clear explanations for its recommendations.
-   **Situation Reporting:** Generates a concise summary of the incident.
-   **REST API:** Exposes its functionality through a FastAPI-based REST API.
-   **Command-Line Interface:** Allows for quick analysis directly from the terminal.
-   **PDF Report Generation:** Can export the analysis into a PDF report.

## Architecture

The project is composed of several key components:

-   **`api`:** A FastAPI server that exposes the core logic as REST endpoints.
-   **`app`:** Contains the main orchestration logic that ties together the different analytical components.
-   **`cli`:** A command-line interface for interacting with the system.
-   **`core`:** The heart of the system, containing modules for:
    -   `classifier.py`: Incident classification.
    -   `explainer.py`: Action explanation generation.
    -   `report.py`: PDF report generation.
    -   `severity.py`: Severity calculation.
    -   `similarity.py`: Recommending actions based on similarity.
    -   `situation.py`: Situation assessment generation.
-   **`frontend`:** A simple web interface for interacting with the API.
-   **`models`:** Stores the trained machine learning models.
-   **`data`:** Contains the datasets used for training and evaluation.
-   **`knowledge`:** Knowledge bases used by the system (e.g., for action recommendations).

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd IRPR
    ```

2.  **Create a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### API Server

To run the API server:

```bash
uvicorn api.server:app --reload
```

The API documentation will be available at `http://127.0.0.1:8000/docs`.

### Command-Line Interface

To use the CLI to analyze an incident:

```bash
python cli/cli.py "your incident description here"
```

### Frontend

The project also includes a basic frontend. To use it, open the `frontend/index.html` file in a web browser. Make sure the API server is running.

## Project Structure
```
├── api/                  # FastAPI application
├── app/                  # Orchestration logic
├── cli/                  # Command-line interface
├── core/                 # Core functionalities (classification, explanation, etc.)
├── data/                 # Datasets
├── evaluation_results/   # Evaluation scripts and results
├── frontend/             # Frontend application
├── knowledge/            # Knowledge bases
├── models/               # Trained models
├── Phases/               # Scripts for different project phases
└── requirements.txt      # Python dependencies
```
