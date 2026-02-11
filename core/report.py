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


# import re
# from reportlab.platypus import (
#     SimpleDocTemplate,
#     Paragraph,
#     Spacer,
#     Table,
#     TableStyle,
#     Flowable,
#     PageBreak,
# )
# from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
# from reportlab.lib.pagesizes import A4
# from reportlab.lib.units import inch
# from reportlab.lib import colors
# from datetime import datetime

# # --- HELPER FUNCTIONS ---


# def clean_markdown(text):
#     """Convert markdown formatting to ReportLab tags"""
#     if not text:
#         return ""
#     # Convert bold
#     text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
#     # Convert italic
#     text = re.sub(r"\*(.*?)\*", r"<i>\1</i>", text)
#     # Remove backticks
#     text = text.replace("`", "")
#     return text


# def parse_list_items(text):
#     """Parse markdown list items and return formatted paragraphs"""
#     if not text:
#         return []

#     lines = text.split('\n')
#     items = []
#     current_item = ""

#     for line in lines:
#         line = line.strip()
#         if line.startswith('* ') or line.startswith('- '):
#             if current_item:
#                 items.append(current_item)
#             current_item = line[2:]  # Remove bullet
#         elif line and current_item:
#             current_item += " " + line
#         elif line and not current_item:
#             items.append(line)

#     if current_item:
#         items.append(current_item)

#     return items


# class RiskMatrix(Flowable):
#     """Custom Flowable to draw a 5x5 Risk Matrix."""

#     def __init__(self, severity_score, confidence_score):
#         Flowable.__init__(self)
#         self.width = 150
#         self.height = 120
#         # Map 0.0-1.0 scores to 1-5 grid indices
#         self.sev_idx = min(int(severity_score * 5), 4)
#         self.conf_idx = min(int(confidence_score * 5), 4)

#     def draw(self):
#         canvas = self.canv
#         cell_w = 25
#         cell_h = 20

#         canvas.setFont("Helvetica-Bold", 7)
#         canvas.setFillColor(colors.white)
#         canvas.drawString(0, 110, "RISK EXPOSURE")

#         for row in range(5):
#             for col in range(5):
#                 # Risk calculation
#                 r = (col + row) / 8.0
#                 cell_color = colors.linearlyInterpolatedColor(
#                     colors.HexColor("#22c55e"),  # Green
#                     colors.HexColor("#ef4444"),  # Red
#                     0,
#                     1,
#                     r,
#                 )

#                 # Highlight current risk position
#                 if col == self.sev_idx and row == self.conf_idx:
#                     canvas.setStrokeColor(colors.white)
#                     canvas.setLineWidth(1.5)
#                 else:
#                     canvas.setStrokeColor(colors.HexColor("#27272a"))
#                     canvas.setLineWidth(0.5)

#                 canvas.setFillColor(cell_color)
#                 canvas.rect(
#                     col * cell_w, row * cell_h, cell_w, cell_h, fill=1, stroke=1
#                 )

#                 # Draw indicator dot for current position
#                 if col == self.sev_idx and row == self.conf_idx:
#                     canvas.setFillColor(colors.white)
#                     canvas.circle(col * cell_w + 12.5, row * cell_h + 10, 3, fill=1)


# # --- MAIN GENERATOR ---


# def generate_report(filename: str, data: dict):
#     doc = SimpleDocTemplate(
#         filename,
#         pagesize=A4,
#         rightMargin=40,
#         leftMargin=40,
#         topMargin=40,
#         bottomMargin=40,
#     )

#     # Theme Colors
#     BG_DARK = colors.HexColor("#09090b")
#     ACCENT_BLUE = colors.HexColor("#3b82f6")
#     BORDER_COLOR = colors.HexColor("#27272a")
#     TEXT_MUTED = colors.HexColor("#a1a1aa")
#     TEXT_MAIN = colors.HexColor("#d4d4d8")

#     # Styles
#     styles = getSampleStyleSheet()

#     body_style = ParagraphStyle(
#         "Body",
#         fontSize=10,
#         textColor=TEXT_MAIN,
#         leading=14,
#         spaceAfter=8
#     )

#     section_style = ParagraphStyle(
#         "Section",
#         fontSize=14,
#         textColor=ACCENT_BLUE,
#         spaceBefore=20,
#         spaceAfter=10,
#         fontName="Helvetica-Bold",
#     )

#     subsection_style = ParagraphStyle(
#         "Subsection",
#         fontSize=11,
#         textColor=colors.white,
#         spaceBefore=12,
#         spaceAfter=6,
#         fontName="Helvetica-Bold",
#     )

#     bullet_style = ParagraphStyle(
#         "Bullet",
#         fontSize=10,
#         textColor=TEXT_MAIN,
#         leading=14,
#         leftIndent=20,
#         spaceAfter=6,
#         bulletIndent=10,
#     )

#     story = []

#     # ============================================================
#     # 1. HEADER WITH RISK MATRIX
#     # ============================================================
#     sev_score = data.get("severity_score", 0.6)
#     conf_score = data.get("classification_confidence", 0.65) / 100  # Convert to 0-1 scale

#     header_content = [
#         [
#             Paragraph(
#                 "<font size='20' color='white'><b>IRPR Analysis Report</b></font><br/>"
#                 f"<font color='#a1a1aa'>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</font>",
#                 body_style,
#             ),
#             RiskMatrix(sev_score, conf_score),
#         ]
#     ]

#     head_table = Table(header_content, colWidths=[3.5 * inch, 2 * inch])
#     head_table.setStyle(
#         TableStyle(
#             [
#                 ("VALIGN", (0, 0), (-1, -1), "TOP"),
#                 ("ALIGN", (1, 0), (1, 0), "RIGHT")
#             ]
#         )
#     )
#     story.append(head_table)
#     story.append(Spacer(1, 0.2 * inch))
#     story.append(
#         Table(
#             [[""]],
#             colWidths=[7 * inch],
#             rowHeights=[1],
#             style=[("LINEBELOW", (0, 0), (-1, -1), 1, BORDER_COLOR)],
#         )
#     )
#     story.append(Spacer(1, 0.3 * inch))

#     # ============================================================
#     # 2. INCIDENT OVERVIEW
#     # ============================================================
#     story.append(Paragraph("Incident Overview", section_style))

#     overview_data = [
#         ["Severity", f"Critical (Score: {data.get('severity', 'N/A')})"],
#         ["Incident Type", data.get("classification", "Unknown")],
#         ["Classification Confidence", f"{data.get('classification_confidence', 0):.2f}%"],
#     ]

#     overview_table = Table(overview_data, colWidths=[2 * inch, 4.5 * inch])
#     overview_table.setStyle(
#         TableStyle(
#             [
#                 ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#18181b")),
#                 ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
#                 ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
#                 ("FONTSIZE", (0, 0), (-1, -1), 10),
#                 ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
#                 ("LEFTPADDING", (0, 0), (-1, -1), 12),
#                 ("RIGHTPADDING", (0, 0), (-1, -1), 12),
#                 ("TOPPADDING", (0, 0), (-1, -1), 8),
#                 ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
#             ]
#         )
#     )
#     story.append(overview_table)
#     story.append(Spacer(1, 0.3 * inch))

#     # ============================================================
#     # 3. SITUATION ASSESSMENT
#     # ============================================================
#     story.append(Paragraph("Situation Assessment", section_style))

#     situation_text = data.get("situation_assessment", "")
#     if situation_text:
#         # Split by numbered sections
#         sections = re.split(r'\n\d+\.\s+', situation_text)

#         if len(sections) > 1:
#             # First part is intro
#             intro = sections[0].strip()
#             if intro:
#                 story.append(Paragraph(clean_markdown(intro), body_style))
#                 story.append(Spacer(1, 0.1 * inch))

#             # Process numbered sections
#             section_titles = re.findall(r'\d+\.\s+([^\n]+)', situation_text)

#             for i, (title, content) in enumerate(zip(section_titles, sections[1:]), 1):
#                 story.append(Paragraph(f"{i}. {title}", subsection_style))

#                 # Parse list items if present
#                 if '*' in content or '-' in content:
#                     items = parse_list_items(content)
#                     for item in items:
#                         story.append(Paragraph(f"• {clean_markdown(item)}", bullet_style))
#                 else:
#                     story.append(Paragraph(clean_markdown(content.strip()), body_style))

#                 story.append(Spacer(1, 0.1 * inch))
#         else:
#             story.append(Paragraph(clean_markdown(situation_text), body_style))

#     story.append(Spacer(1, 0.2 * inch))

#     # ============================================================
#     # 4. RESPONSE PLAN
#     # ============================================================
#     story.append(Paragraph("Response Plan Summary", section_style))
#     story.append(
#         Paragraph(
#             "Strategic remediation steps based on incident vectors:",
#             body_style
#         )
#     )
#     story.append(Spacer(1, 0.1 * inch))

#     # Response Plan Table
#     plan_data = [["ID", "Action Phase", "Confidence"]]
#     actions = data.get("actions", [])

#     for action in actions:
#         plan_data.append(
#             [
#                 action.get("action_id", "N/A"),
#                 action.get("phase", "N/A"),
#                 f"{action.get('confidence', 0):.1f}%"
#             ]
#         )

#     plan_table = Table(plan_data, colWidths=[1.2 * inch, 3.5 * inch, 1.8 * inch])
#     plan_table.setStyle(
#         TableStyle(
#             [
#                 ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#18181b")),
#                 ("TEXTCOLOR", (0, 0), (-1, 0), TEXT_MUTED),
#                 ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
#                 ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
#                 ("FONTSIZE", (0, 0), (-1, -1), 9),
#                 ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.transparent, colors.HexColor("#0f172a")]),
#                 ("TEXTCOLOR", (0, 1), (-1, -1), colors.white),
#                 ("LEFTPADDING", (0, 0), (-1, -1), 10),
#                 ("RIGHTPADDING", (0, 0), (-1, -1), 10),
#                 ("TOPPADDING", (0, 0), (-1, -1), 8),
#                 ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
#                 ("ALIGN", (2, 0), (2, -1), "CENTER"),
#             ]
#         )
#     )
#     story.append(plan_table)
#     story.append(Spacer(1, 0.3 * inch))

#     # ============================================================
#     # 5. ACTION RATIONALE (New Section)
#     # ============================================================
#     story.append(PageBreak())
#     story.append(Paragraph("Action Rationale", section_style))
#     story.append(
#         Paragraph(
#             "Detailed reasoning for each recommended response action:",
#             body_style
#         )
#     )
#     story.append(Spacer(1, 0.2 * inch))

#     # Get rationale data
#     rationale_data = data.get("rationale", {})

#     for action in actions:
#         action_id = action.get("action_id", "")
#         rationale_text = rationale_data.get(action_id, "No rationale provided.")

#         # Action ID header
#         action_header_data = [[
#             Paragraph(f"<b>{action_id}</b>", body_style),
#             Paragraph(f"<font color='#a1a1aa'>{action.get('phase', 'N/A')}</font>", body_style)
#         ]]

#         action_header_table = Table(action_header_data, colWidths=[2 * inch, 4.5 * inch])
#         action_header_table.setStyle(
#             TableStyle([
#                 ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#18181b")),
#                 ("LEFTPADDING", (0, 0), (-1, -1), 12),
#                 ("RIGHTPADDING", (0, 0), (-1, -1), 12),
#                 ("TOPPADDING", (0, 0), (-1, -1), 8),
#                 ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
#             ])
#         )
#         story.append(action_header_table)

#         # Rationale content
#         rationale_content_table = Table(
#             [[Paragraph(clean_markdown(rationale_text), body_style)]],
#             colWidths=[6.5 * inch]
#         )
#         rationale_content_table.setStyle(
#             TableStyle([
#                 ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#0f172a")),
#                 ("BOX", (0, 0), (-1, -1), 1, BORDER_COLOR),
#                 ("LEFTPADDING", (0, 0), (-1, -1), 12),
#                 ("RIGHTPADDING", (0, 0), (-1, -1), 12),
#                 ("TOPPADDING", (0, 0), (-1, -1), 10),
#                 ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
#             ])
#         )
#         story.append(rationale_content_table)
#         story.append(Spacer(1, 0.15 * inch))

#     # ============================================================
#     # 6. EVIDENCE & HISTORICAL CONTEXT (New Section)
#     # ============================================================
#     story.append(PageBreak())
#     story.append(Paragraph("Evidence & Historical Context", section_style))
#     story.append(
#         Paragraph(
#             "Similar historical incidents that informed this analysis:",
#             body_style
#         )
#     )
#     story.append(Spacer(1, 0.2 * inch))

#     # Get evidence data
#     evidence = data.get("evidence", [])

#     if evidence:
#         for i, incident in enumerate(evidence, 1):
#             incident_type = incident.get("incident_type", "Unknown")
#             similarity = incident.get("similarity_score", 0)
#             description = incident.get("description", "No description available.")

#             # Incident header
#             header_text = f"<b>({i}) [{incident_type}]</b> <font color='#22c55e'>Similarity: {similarity:.3f}</font>"
#             story.append(Paragraph(header_text, body_style))

#             # Incident description in bordered box
#             incident_table = Table(
#                 [[Paragraph(clean_markdown(description), body_style)]],
#                 colWidths=[6.5 * inch]
#             )
#             incident_table.setStyle(
#                 TableStyle([
#                     ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#0f172a")),
#                     ("BOX", (0, 0), (-1, -1), 1, BORDER_COLOR),
#                     ("LEFTPADDING", (0, 0), (-1, -1), 12),
#                     ("RIGHTPADDING", (0, 0), (-1, -1), 12),
#                     ("TOPPADDING", (0, 0), (-1, -1), 10),
#                     ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
#                 ])
#             )
#             story.append(incident_table)
#             story.append(Spacer(1, 0.15 * inch))
#     else:
#         story.append(
#             Paragraph(
#                 "<i>No historical evidence data available.</i>",
#                 body_style
#             )
#         )

#     # ============================================================
#     # 7. ANALYST NOTES (Optional)
#     # ============================================================
#     if data.get("analyst_notes"):
#         story.append(PageBreak())
#         story.append(Paragraph("Analyst Override & Manual Notes", section_style))

#         notes_table = Table(
#             [[Paragraph(clean_markdown(data["analyst_notes"]), body_style)]],
#             colWidths=[6.5 * inch]
#         )
#         notes_table.setStyle(
#             TableStyle([
#                 ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#1e1e2e")),
#                 ("BOX", (0, 0), (-1, -1), 2, ACCENT_BLUE),
#                 ("LEFTPADDING", (0, 0), (-1, -1), 15),
#                 ("RIGHTPADDING", (0, 0), (-1, -1), 15),
#                 ("TOPPADDING", (0, 0), (-1, -1), 15),
#                 ("BOTTOMPADDING", (0, 0), (-1, -1), 15),
#             ])
#         )
#         story.append(notes_table)

#     # ============================================================
#     # 8. BUILD PDF
#     # ============================================================
#     def on_page(canvas, doc):
#         """Apply dark background to all pages"""
#         canvas.saveState()
#         canvas.setFillColor(BG_DARK)
#         canvas.rect(0, 0, A4[0], A4[1], fill=1)

#         # Add page number
#         canvas.setFont("Helvetica", 8)
#         canvas.setFillColor(TEXT_MUTED)
#         page_num = canvas.getPageNumber()
#         text = f"Page {page_num}"
#         canvas.drawRightString(A4[0] - 40, 20, text)

#         canvas.restoreState()

#     doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
#     print(f"✓ Report generated: {filename}")
