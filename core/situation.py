from groq import Groq

client = Groq()

def generate_situation_assessment(
    incident_text: str,
    incident_type: str,
    confidence: float,
    similar_incidents: list
) -> str:
    
    # Summarize similar incidents for grounding
    similar_summary = "\n".join(
        f"- [{s['incident_type']}] similarity={s['similarity']:.2f}"
        for s in similar_incidents[:3]
    )



    prompt = f"""
        You are generating a professional incident response situation assessment
        for a security analyst.

        Incident description:
        {incident_text}
        Model assessment:
        - Predicted incident type: {incident_type}
        - Classification confidence: {confidence:.2f}

        Operational context:
        - Activity occurred outside normal business hours
        - Legitimate system activity is expected to be minimal

        Similar historical incidents indicate:
        {similar_summary}

        Write a structured situation assessment with the following sections:

        1. Why this activity is noteworthy
        2. Key risk factors in this context
        3. Factors that limit certainty or reduce risk
        4. What assumptions should NOT be made yet

        Constraints:
        - Do NOT recommend actions
        - Do NOT provide procedural steps
        - Use professional, report-style language
    """

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        # max_tokens=350
    )

    return completion.choices[0].message.content.strip()
