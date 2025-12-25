# validates if the ACTION_MAP is aligned with Action_kb.json


def validate_action_kb(action_kb: dict, action_map: dict):
    """
    Ensure all action IDs used by the system exist in the action knowledge base.
    """
    missing = set()

    for incident_type, actions in action_map.items():
        for action_id in actions:
            if action_id not in action_kb:
                missing.add(action_id)

    if missing:
        raise RuntimeError(
            "Action KB validation failed. Missing action IDs: "
            + ", ".join(sorted(missing))
        )
