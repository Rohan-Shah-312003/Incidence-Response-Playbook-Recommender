import sys
from app.orchestrator import run_pipeline

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: irpr \"incident description\"")
        sys.exit(1)

    incident = sys.argv[1]
    output = run_pipeline(incident)

    print("\nIncident Type:", output["incident_type"])
    print("\nRecommended Actions:\n")

    for action, explanation in output["explanations"].items():
        print("="*50)
        print(explanation)
