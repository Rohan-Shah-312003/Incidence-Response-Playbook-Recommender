import re
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Flowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from datetime import datetime

# --- HELPER FUNCTIONS ---


def clean_markdown(text):
    if not text:
        return ""
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.*?)\*", r"<i>\1</i>", text)
    return text.replace("`", "")


class RiskMatrix(Flowable):
    """Custom Flowable to draw a 5x5 Risk Matrix."""

    def __init__(self, severity_score, confidence_score):
        Flowable.__init__(self)
        self.width = 150
        self.height = 120
        # Map 0.0-1.0 scores to 1-5 grid indices
        self.sev_idx = min(int(severity_score * 5), 4)
        self.conf_idx = min(int(confidence_score * 5), 4)

    def draw(self):
        canvas = self.canv
        cell_w = 25
        cell_h = 20

        canvas.setFont("Helvetica-Bold", 7)
        canvas.setFillColor(colors.white)
        canvas.drawString(0, 110, "RISK EXPOSURE")

        for row in range(5):
            for col in range(5):
                # Corrected interpolation function name
                # interp(color1, color2, low, high, value)
                r = (col + row) / 8.0
                cell_color = colors.linearlyInterpolatedColor(
                    colors.HexColor("#22c55e"),  # Green
                    colors.HexColor("#ef4444"),  # Red
                    0,
                    1,
                    r,
                )

                # Highlight logic
                if col == self.sev_idx and row == self.conf_idx:
                    canvas.setStrokeColor(colors.white)
                    canvas.setLineWidth(1.5)
                else:
                    canvas.setStrokeColor(colors.HexColor("#27272a"))
                    canvas.setLineWidth(0.5)

                canvas.setFillColor(cell_color)
                canvas.rect(
                    col * cell_w, row * cell_h, cell_w, cell_h, fill=1, stroke=1
                )

                # Draw indicator dot
                if col == self.sev_idx and row == self.conf_idx:
                    canvas.setFillColor(colors.white)
                    canvas.circle(col * cell_w + 12.5, row * cell_h + 10, 3, fill=1)


# --- MAIN GENERATOR ---


def generate_report(filename: str, data: dict):
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    # Theme Colors
    BG_DARK = colors.HexColor("#09090b")
    ACCENT_BLUE = colors.HexColor("#3b82f6")
    BORDER_COLOR = colors.HexColor("#27272a")

    styles = getSampleStyleSheet()
    body_style = ParagraphStyle(
        "Body", fontSize=10, textColor=colors.HexColor("#d4d4d8"), leading=14
    )
    section_style = ParagraphStyle(
        "Section",
        fontSize=14,
        textColor=ACCENT_BLUE,
        spaceBefore=20,
        fontName="Helvetica-Bold",
    )

    story = []

    # 1. HEADER WITH RISK MATRIX
    # We use a score from data. Defaulting to 0.7 if not found for visual demo.
    sev_score = data.get("severity_score", 0.6)
    conf_score = data.get("classification_confidence", 0.8)

    header_content = [
        [
            Paragraph(
                "<font size='20' color='white'><b>IRPR Analysis Report</b></font><br/>"
                f"<font color='#a1a1aa'>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</font>",
                body_style,
            ),
            RiskMatrix(sev_score, conf_score),
        ]
    ]

    head_table = Table(header_content, colWidths=[3.5 * inch, 2 * inch])
    head_table.setStyle(
        TableStyle(
            [("VALIGN", (0, 0), (-1, -1), "TOP"), ("ALIGN", (1, 0), (1, 0), "RIGHT")]
        )
    )
    story.append(head_table)
    story.append(Spacer(1, 0.2 * inch))
    story.append(
        Table(
            [[""]],
            colWidths=[7 * inch],
            rowHeights=[1],
            style=[("LINEBELOW", (0, 0), (-1, -1), 1, BORDER_COLOR)],
        )
    )

    # 2. DATA SECTIONS
    sections = [
        ("Situation Assessment", data["situation_assessment"]),
        (
            "Response Plan Summary",
            "Strategic remediation steps based on incident vectors:",
        ),
    ]

    for title, text in sections:
        story.append(Paragraph(title, section_style))
        story.append(Paragraph(clean_markdown(text).replace("\n", "<br/>"), body_style))

    # 3. RESPONSE PLAN TABLE
    plan_data = [["ID", "Action Phase", "Confidence"]]
    for action in data["actions"]:
        plan_data.append(
            [action["action_id"], action["phase"], f"{round(action['confidence'], 1)}%"]
        )

    t = Table(plan_data, colWidths=[1 * inch, 3 * inch, 1.5 * inch])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#18181b")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#a1a1aa")),
                ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.transparent, colors.HexColor("#0f172a")],
                ),
                ("TEXTCOLOR", (0, 1), (-1, -1), colors.white),
            ]
        )
    )
    story.append(Spacer(1, 0.2 * inch))
    story.append(t)

    # 4. BACKGROUND
    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(BG_DARK)
        canvas.rect(0, 0, A4[0], A4[1], fill=1)
        canvas.restoreState()

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    if data.get("analyst_notes"):
        story.append(Paragraph("Analyst Override & Manual Notes", section_style))
        # Use a distinct background for manual notes to separate them from AI output
        notes_table = Table(
            [[Paragraph(data["analyst_notes"], body_style)]], colWidths=[6 * inch]
        )
        notes_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#1e1e2e")),
                    ("BOX", (0, 0), (-1, -1), 1, ACCENT_BLUE),
                    ("LEFTPADDING", (0, 0), (-1, -1), 15),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 15),
                    ("TOPPADDING", (0, 0), (-1, -1), 15),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 15),
                ]
            )
        )
        story.append(notes_table)
