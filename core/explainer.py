import json
from groq import Groq
from dotenv import load_dotenv
load_dotenv()


client = Groq()

with open("knowledge/action_kb.json") as f:
    ACTION_KB = json.load(f)

# from core.action_kb import ACTION_KB

def explain_actions(incident_text: str, actions: list):
    explanations = {}

    for action in actions:
        action_id = action["action_id"]
        kb = ACTION_KB.get(action_id)

        if not kb:
            continue

        explanations[action_id] = (
            f"{kb['description']} "
            f"This action is recommended during the {kb['phase']} phase "
            f"to reduce risk. If skipped, {kb['risk_if_skipped']}."
        )

        # explanations = explain_actions(incident_text, actions)

        result = {
            "actions": actions,
            "explanations": explanations,
            
        }

    return explanations
    # return result
