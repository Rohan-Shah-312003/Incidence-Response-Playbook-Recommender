import json
from groq import Groq

client = Groq()

with open("knowledge/action_kb.json") as f:
    ACTION_KB = json.load(f)

def explain_actions(incident_text: str, actions: list):
    explanations = {}

    for action in actions:
        action_id = action["action_id"]
        confidence = action["confidence"]

        if action_id not in ACTION_KB:
            raise RuntimeError(
                f"Action '{action_id}' missing from action_kb.json"
            )

        action_meta = ACTION_KB[action_id]

        prompt = f"""
            You are assisting a security analyst.

            Constraints:
            - ONLY explain the provided response action
            - Do NOT suggest new actions
            - Do NOT invent procedures or tools

            Incident context:
            {incident_text}

            Model assessment:
            Predicted incident type: {action_meta['phase']}
            Classification confidence: {confidence:.2f}

            Operational context:
            - Activity occurred outside normal business hours
            - Legitimate system usage is expected to be low
            - Monitoring and staffing may be reduced

            Action under explanation:
            Action name: {action_meta['action_name']}
            Response phase: {action_meta['phase']}

            Action knowledge:
            Description:
            {action_meta['description']}

            Implementation guidance:
            {action_meta['implementation_guidance']}

            Risk if skipped:
            {action_meta['risk_if_skipped']}

            Task:
            Explain clearly:
            1. Why this action is relevant for this incident
            2. Why it should be performed at this stage of the response
            3. Why delaying this action could increase risk in this context

            Keep the explanation concise, professional, and cautious.
        """


        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            # max_tokens=300
        )

        explanations[action_id] = completion.choices[0].message.content.strip()

    return explanations
