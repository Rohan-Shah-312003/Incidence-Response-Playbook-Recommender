from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch


def generate_report(filename: str, data: dict):
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50,
    )

    styles = getSampleStyleSheet()
    story = []

    def add(text):
        story.append(Paragraph(text, styles["Normal"]))
        story.append(Spacer(1, 0.15 * inch))

    # Title
    story.append(Paragraph("<b>Incident Response Report</b>", styles["Title"]))
    story.append(Spacer(1, 0.3 * inch))

    # Metadata
    add(f"<b>Incident Type:</b> {data['incident_type']}")
    add(f"<b>Severity:</b> {data['severity']['level']}")
    add(f"<b>Confidence:</b> {round(data['classification_confidence'], 2)}")

    story.append(Spacer(1, 0.3 * inch))

    # Situation
    story.append(Paragraph("<b>Situation Assessment</b>", styles["Heading2"]))
    add(data["situation_assessment"].replace("\n", "<br/>"))

    # Response Plan
    story.append(Paragraph("<b>Response Plan</b>", styles["Heading2"]))
    for action in data["actions"]:
        add(
            f"<b>{action['action_id']}</b> ({action['phase']}) "
            f"- Relevance: {round(action['confidence'], 1)}%"
        )

    # Evidence
    story.append(Paragraph("<b>Evidence</b>", styles["Heading2"]))
    for e in data["similar_incidents"]:
        add(
            f"[{e['incident_type']}] similarity={round(e['similarity'], 3)}<br/>"
            f"{e['text']}"
        )

    doc.build(story)
