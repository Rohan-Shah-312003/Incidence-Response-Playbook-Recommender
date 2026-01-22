function openTab(tabId) {
	document
		.querySelectorAll(".tab-content")
		.forEach((t) => t.classList.remove("active"));
	document
		.querySelectorAll(".tab-btn")
		.forEach((b) => b.classList.remove("active"));
	document.getElementById(tabId).classList.add("active");
	event.target.classList.add("active");
}

async function analyze() {
	const input = document.getElementById("incidentInput").value;
	if (!input.trim()) {
		alert("Please enter an incident description.");
		return;
	}

	openTab("situation");

	document.getElementById("situation").innerText = "Analyzing incident...";
	document.getElementById("plan").innerText = "";
	document.getElementById("rationale").innerText = "";
	document.getElementById("evidence").innerText = "";

	try {
		const data = await window.api.analyzeIncident(input);

		/* -------- SITUATION -------- */
		document.getElementById("situation").innerText =
			`Severity: ${data.severity.level} (Score: ${data.severity.score})\n\n` +
			`Incident Type: ${data.incident_type}\n` +
			`Classification Confidence: ${(
				data.classification_confidence * 100
			).toFixed(2)}%\n\n` +
			data.situation_assessment;

		/* -------- RESPONSE PLAN -------- */
		let planText = "";
		data.actions.forEach((a, idx) => {
			planText += `${idx + 1}. ${a.action_id} (${a.phase})\n`;
			planText += `   Relative relevance score: ${(
				a.confidence * 100
			).toFixed(1)}%\n\n`;
		});
		document.getElementById("plan").innerText = planText;

		/* -------- RATIONALE -------- */
		let rationaleText = "";
		data.actions.forEach((a) => {
			rationaleText += "----------------------------------------\n";
			rationaleText += `Action: ${a.action_id}\n\n`;
			rationaleText += data.explanations[a.action_id] + "\n\n";
			const explanation = rationaleText;
			if (explanation) {
				rationaleText += explanation + "\n\n";
			} else {
				rationaleText +=
					"No explanation available for this action.\n\n";
			}
		});
		document.getElementById("rationale").innerText = rationaleText;

		/* -------- EVIDENCE -------- */
		let evidenceText = "Similar historical incidents:\n\n";
		data.similar_incidents.forEach((item, idx) => {
			evidenceText += `(${idx + 1}) [${item.incident_type}] `;
			evidenceText += `similarity=${item.similarity.toFixed(3)}\n`;
			evidenceText += `${item.text}\n\n`;
		});
		document.getElementById("evidence").innerText = evidenceText;
		async function submitOverride() {
			const note = document.getElementById("overrideNote").value;

			await fetch("http://127.0.0.1:8000/override", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					corrected_incident_type: "Manual Review",
					analyst_note: note,
				}),
			});

			alert("Analyst override recorded.");
		}
	} catch (err) {
		document.getElementById("situation").innerText =
			"Error communicating with backend.";
	}
}

async function exportReport() {
  await fetch("http://127.0.0.1:8000/export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      incident_text: document.getElementById("incidentInput").value
    })
  });

  alert("Incident report generated.");
}



async function submitOverride() {
	const note = document.getElementById("overrideNote").value;

	await fetch("http://127.0.0.1:8000/override", {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({
			corrected_incident_type: "Manual Review",
			analyst_note: note,
		}),
	});

	alert("Analyst override recorded.");
}
