from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.orchestrator import run_pipeline
from core.report import generate_report
import os
import sqlite3
from datetime import datetime

app = FastAPI(
    title="IRPR – Incident Response Decision Engine",
    description="ML + Similarity + RAG based incident response assistant",
    version="1.0.0",
)

# -------------------------
# Database setup
# -------------------------
DB_PATH = os.path.join(os.path.expanduser("~/Desktop"), "irpr_overrides.db")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS overrides (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            incident_text TEXT NOT NULL,
            corrected_incident_type TEXT NOT NULL,
            analyst_note TEXT
        )
    """)
    conn.commit()
    conn.close()


@app.on_event("startup")
async def startup_event():
    init_db()


# -------------------------
# Request schema
# -------------------------


class IncidentRequest(BaseModel):
    incident_text: str


# -------------------------
# Response schema
# -------------------------


class SimilarIncident(BaseModel):
    incident_type: str
    similarity: float
    text: str


class Severity(BaseModel):
    level: str
    score: int


class AnalystOverride(BaseModel):
    incident_text: str
    corrected_incident_type: str
    analyst_note: str


class ActionExplanation(BaseModel):
    action_id: str
    explanation: str


class ActionItem(BaseModel):
    action_id: str
    confidence: float
    phase: str


class IncidentResponse(BaseModel):
    incident_type: str
    classification_confidence: float
    situation_assessment: str
    actions: list[ActionItem]
    explanations: list[ActionExplanation]
    similar_incidents: list[SimilarIncident]
    severity: Severity


# -------------------------
# API Endpoint
# -------------------------


@app.post("/export")
def export_report(req: IncidentRequest):
    result = run_pipeline(req.incident_text)
    desktop_path = os.path.expanduser("~/Desktop")
    filename = os.path.join(desktop_path, "incident_report.pdf")
    generate_report(filename, result)
    return {"status": "report generated", "file": filename}


@app.post("/override")
def analyst_override(override: AnalystOverride):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO overrides (incident_text, corrected_incident_type, analyst_note) VALUES (?, ?, ?)",
            (
                override.incident_text,
                override.corrected_incident_type,
                override.analyst_note,
            ),
        )
        conn.commit()
        conn.close()
        return {"status": "override recorded"}
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")


@app.post("/analyze", response_model=IncidentResponse)
def analyze_incident(req: IncidentRequest):
    if not req.incident_text.strip():
        raise HTTPException(status_code=400, detail="Incident text cannot be empty")

    result = run_pipeline(req.incident_text)

    return {
        "incident_type": result["incident_type"],
        "classification_confidence": result["classification_confidence"],
        "actions": [
            {
                "action_id": a["action_id"],
                "confidence": a["confidence"],
                "phase": a["phase"],
            }
            for a in result["actions"]
        ],
        "explanations": [
            {"action_id": aid, "explanation": text}
            for aid, text in result["explanations"].items()
        ],
        "similar_incidents": result["similar_incidents"],
        "situation_assessment": result["situation_assessment"],
        "severity": result["severity"],
    }
