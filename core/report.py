from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

def generate_report(filename: str, result: dict):
    c = canvas.Canvas(filename, pagesize=A4)
    text = c.beginText(40, 800)

    def line(s):
        text.textLine(s)

    line("Incident Response Report")
    line("=" * 50)
    line("")
    line(f"Incident Type: {result['incident_type']}")
    line(f"Severity: {result['severity']['level']}")
    line(f"Confidence: {result['classification_confidence']:.2f}")
    line("")

    line("Situation Assessment:")
    for l in result["situation_assessment"].split("\n"):
        line(l)

    line("")
    line("Response Plan:")
    for a in result["actions"]:
        line(f"- {a['action_id']} ({a['phase']})")

    line("")
    line("Evidence:")
    for e in result["similar_incidents"]:
        line(f"[{e['incident_type']}] sim={e['similarity']:.2f}")
        line(e["text"][:300])
        line("")

    c.drawText(text)
    c.save()
