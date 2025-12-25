async function analyze() {
  const input = document.getElementById("incidentInput").value;
  const output = document.getElementById("output");

  if (!input.trim()) {
    output.innerText = "Please enter an incident description.";
    return;
  }

  output.innerText = "Analyzing incident...\n";

  try {
    const data = await window.api.analyzeIncident(input);

    let text = "";
    text += `Incident Type: ${data.incident_type}\n`;
    text += `Confidence: ${(data.classification_confidence * 100).toFixed(2)}%\n\n`;

    data.explanations.forEach(item => {
      text += "----------------------------------------\n";
      text += `Action: ${item.action_id}\n\n`;
      text += item.explanation + "\n\n";
    });

    output.innerText = text;

  } catch (err) {
    output.innerText = "Error communicating with backend.";
  }
}
