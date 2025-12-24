import os
from dotenv import load_dotenv
from groq import Groq



load_dotenv()
client = Groq()

def explain_action_llm(
    incident_text: str,
    action_id: str,
    confidence: float,
    action_kb: dict
) -> str:
    action = action_kb[action_id]

    prompt = f"""
        You are an incident response assistant.

        You must ONLY explain the provided response action.
        Do NOT suggest new actions.
        Do NOT invent procedures.
        Do NOT change the scope.

        Incident description:
        {incident_text}

        Recommended response action:
        Action ID: {action_id}
        Action name: {action['action_name']}
        Confidence score: {confidence:.2f}

        Action knowledge:
        Description:
        {action['description']}

        Implementation guidance:
        {action['implementation_guidance']}

        Risk if skipped:
        {action['risk_if_skipped']}

        Task:
        Explain, in clear professional language:
        1. Why this action is relevant to the incident
        2. How an analyst should execute it
        3. What risk this action mitigates

        Keep the explanation concise and factual.
    """

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,   # 🔑 LOW temperature
        max_tokens=250     # Enough for explanation, not rambling
    )

    return completion.choices[0].message.content.strip()
