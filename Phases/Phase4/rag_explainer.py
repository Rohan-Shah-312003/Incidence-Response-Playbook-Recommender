import json
from llm_explainer import explain_action_llm

def explain_action(action_id, confidence, incident_text, action_kb):
    for action_id, conf in ranked_actions:
        explanation = explain_action_llm(
            incident_text,
            action_id,
            conf,
            action_kb
        )
        print("=" * 60)
        print(explanation)
    

    action = action_kb[action_id]

    explanation = f"""
        Action: {action["action_name"]}
        Phase: {action["phase"]}
        Confidence: {confidence:.2f}

        Why this action matters:
        This action is recommended because the incident involves behaviors consistent with
        {incident_text[:120]}..., which aligns with the need to {action["description"].lower()}.

        How to execute:
        {action["implementation_guidance"]}

        Risk if skipped:
        {action["risk_if_skipped"]}
        """

    return explanation.strip()


if __name__ == "__main__":

    with open("./data/action_kb.json") as f:
        action_kb = json.load(f)

    # Example input from Phase 3
    # incident_text = (
    #     "An employee received an urgent email requesting credential verification."
    # )

    incident_text = (
    """
        An admin mentioned facing problems in logging in their employee-only portal.
        The admin password was only known by one of their close friends.
    """
    )

    ranked_actions = [("IR-ID-01", 0.33), ("IR-CON-02", 0.33)]

    for action_id, conf in ranked_actions:
        print("=" * 50)
        print(explain_action(action_id, conf, incident_text, action_kb))
