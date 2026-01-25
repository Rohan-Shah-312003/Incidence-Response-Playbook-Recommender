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

    for action in actions[:5]:  # Limit to top 5 to save API calls
        action_id = action.get("action_id")
        if not action_id or action_id not in ACTION_KB:
            continue

        kb = ACTION_KB[action_id]
        
        # Enhanced prompt with few-shot example
        prompt = f"""You are an incident response expert. Explain why this action is relevant.

**Incident Description:**
{incident_text}

**Recommended Action:**
- ID: {action_id}
- Name: {kb['action_name']}
- Phase: {kb['phase']}
- Relevance Score: {action['confidence']:.1f}%

**Action Details:**
Description: {kb['description']}
Implementation: {kb['implementation_guidance']}
Risk if skipped: {kb['risk_if_skipped']}

**Your Task:**
Write a professional 2-3 sentence explanation covering:
1. Why this action is relevant to THIS specific incident
2. What the analyst should do
3. What risk this mitigates

**Example:**
For a phishing incident, "Disable compromised account" would be explained as:
"This action prevents further unauthorized access using the exposed credentials from the phishing attack. The analyst should immediately disable the affected user account in the identity management system and invalidate all active sessions. If skipped, the attacker could continue accessing email, applications, and potentially pivot to other systems."

**Your explanation (2-3 sentences):**"""

        try:
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,  # Low temperature for consistency
                max_tokens=200
            )
            
            explanation = completion.choices[0].message.content.strip()
            
            # Validate response
            if len(explanation) < 50:
                # Fallback to template
                explanation = _template_explanation(kb, action['confidence'])
                
            explanations[action_id] = explanation
            
        except Exception as e:
            print(f"Error generating explanation for {action_id}: {e}")
            # Fallback to template
            explanations[action_id] = _template_explanation(kb, action['confidence'])

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