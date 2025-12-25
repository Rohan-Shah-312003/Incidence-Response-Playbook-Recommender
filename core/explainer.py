import json
from groq import Groq
from core.validator import validate_action_kb
from app.config import ENABLE_LLM
from core.similarity import ACTION_MAP


client = Groq()

with open("./knowledge/action_kb.json") as f:
    ACTION_KB = json.load(f)

validate_action_kb(ACTION_KB, ACTION_MAP)



def explain_actions(incident_text: str, actions):
    explanations = {}

    for action_id, confidence in actions:
        action = ACTION_KB[action_id]
        if not ENABLE_LLM:
            return f"[LLM disabled] Action: {action['action_name']}"

        prompt = f"""
You are an incident response assistant.

ONLY explain the provided action.

Incident:
{incident_text}

Action:
{action["action_name"]} (confidence {confidence:.2f})

Description:
{action["description"]}

Implementation guidance:
{action["implementation_guidance"]}

Risk if skipped:
{action["risk_if_skipped"]}
"""

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=250,
        )

        explanations[action_id] = completion.choices[0].message.content.strip()

    return explanations
