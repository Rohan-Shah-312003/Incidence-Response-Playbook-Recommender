import json
from groq import Groq
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()


client = Groq()

KB_PATH = Path(__file__).resolve().parent.parent / "knowledge" / "action_kb.json"

with open(KB_PATH) as f:
    ACTION_KB = json.load(f)


def explain_actions(incident_text: str, actions: list):
    explanations = {}

    for action in actions:
        action_id = action.get("action_id")
        if not action_id:
            continue

        kb = ACTION_KB.get(action_id)
        if not kb:
            continue

        explanations[action_id] = (
            f"{kb['description']} "
            f"This action is performed during the {kb['phase']} phase. "
            f"Implementation guidance: {kb['implementation_guidance']} "
            f"If skipped, {kb['risk_if_skipped']}."
        )

    return explanations
