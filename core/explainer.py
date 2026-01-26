"""
Enhanced explainer with better prompt engineering
"""

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
    """
    Generate explanations for recommended actions

    Args:
        incident_text: Incident description
        actions: List of recommended actions

    Returns:
        Dict mapping action_id to explanation
    """
    explanations = {}

    for action in actions[:20]:
        action_id = action.get("action_id")
        if not action_id or action_id not in ACTION_KB:
            continue

        kb = ACTION_KB[action_id]

        prompt = f"""
        You are an incident response expert. Explain why this action is relevant.

**Incident Description:**
{incident_text}

**Recommended Action:**
- ID: {action_id}
- Name: {kb["action_name"]}
- Phase: {kb["phase"]}

**Action Details:**
Description: {kb["description"]}
Risk if skipped: {kb["risk_if_skipped"]}

**Your Task:**
Write a professional 2-3 sentence explanation.

**Constraints:**
1. DO NOT include a heading, title, or label (e.g., do not write "Action Explanation:" or "Rationale:").
2. DO NOT repeat the Action Name or ID.
3. Start the sentence immediately with the explanation.

**Example:**
"This action prevents further unauthorized access using the exposed credentials. The analyst should immediately disable the affected user account and invalidate active sessions. If skipped, the attacker could continue accessing email and pivoting to other systems."

**Your explanation (2-3 sentences):**"""

        try:
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,  # Low temperature for consistency
                max_tokens=200,
            )

            explanation = completion.choices[0].message.content.strip()

            # Validate response
            if len(explanation) < 50:
                # Fallback to template
                explanation = _template_explanation(kb, action["confidence"])

            explanations[action_id] = explanation

        except Exception as e:
            print(f"Error generating explanation for {action_id}: {e}")
            # Fallback to template
            explanations[action_id] = _template_explanation(kb, action["confidence"])

    return explanations


def _template_explanation(kb: dict, confidence: float) -> str:
    """Fallback template explanation"""
    return (
        f"{kb['description']} "
        f"This action should be performed during the {kb['phase']} phase. "
        f"Implementation: {kb['implementation_guidance']} "
        f"If skipped, {kb['risk_if_skipped']} "
        f"(Relevance: {confidence:.1f}%)"
    )
