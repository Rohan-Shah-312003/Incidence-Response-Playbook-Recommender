// // function renderMarkdown(text) {
// //   if (window.markdown && window.markdown.render) {
// //     return window.markdown.render(text);
// //   }
// //   // Fallback: plain text
// //   return `<pre>${text}</pre>`;
// // }

// // function openTab(tabId, evt) {
// // 	document
// // 		.querySelectorAll(".tab-content")
// // 		.forEach((t) => t.classList.remove("active"));
// // 	document
// // 		.querySelectorAll(".tab-btn")
// // 		.forEach((b) => b.classList.remove("active"));

// // 	document.getElementById(tabId).classList.add("active");
// // 	if (evt) evt.target.classList.add("active");
// // }

// // async function analyze() {
// // 	const input = document.getElementById("incidentInput").value;
// // 	if (!input.trim()) {
// // 		alert("Please enter an incident description.");
// // 		return;
// // 	}

// // 	openTab("situation", event);

// // 	document.getElementById("situation").innerText = "Analyzing incident...";
// // 	document.getElementById("plan").innerText = "";
// // 	document.getElementById("rationale").innerText = "";
// // 	document.getElementById("evidence").innerText = "";

// // 	try {
// // 		const data = await window.api.analyzeIncident(input);
// // 		/* -------- SITUATION -------- */
// // 		document.getElementById("situation").innerText =
// // 			`Severity: ${data.severity.level} (Score: ${data.severity.score})\n\n` +
// // 			`Incident Type: ${data.incident_type}\n` +
// // 			`Classification Confidence: ${(
// // 				data.classification_confidence * 100
// // 			).toFixed(2)}%\n\n` +
// // 			data.situation_assessment;

// // 		/* -------- RESPONSE PLAN -------- */
// // 		let planText = "";
// // 		data.actions.forEach((a, idx) => {
// // 			planText += `${idx + 1}. ${a.action_id} (${a.phase})\n`;
// // 			planText += `   Relative relevance score: ${a.confidence.toFixed(
// // 				1,
// // 			)}%\n\n`;
// // 		});
// // 		document.getElementById("plan").innerText = planText;

// // 		/* -------- RATIONALE -------- */
// // 		let rationaleText = "";

// // 		// Normalize explanations into a map
// // 		let explanationMap = {};

// // 		if (Array.isArray(data.explanations)) {
// // 			data.explanations.forEach((e) => {
// // 				explanationMap[e.action_id] = e.explanation;
// // 			});
// // 		} else if (typeof data.explanations === "string") {
// // 			try {
// // 				const parsed = JSON.parse(data.explanations);
// // 				parsed.forEach((e) => {
// // 					explanationMap[e.action_id] = e.explanation;
// // 				});
// // 			} catch (err) {
// // 				console.error("Failed to parse explanations:", err);
// // 			}
// // 		}

// // 		data.actions.forEach((action) => {
// // 			rationaleText += "----------------------------------------\n";
// // 			rationaleText += `Action: ${action.action_id}\n\n`;
// // 			rationaleText += explanationMap[action.action_id]
// // 				? explanationMap[action.action_id]
// // 				: "No explanation available.\n";
// // 			rationaleText += "\n\n";
// // 		});

// // 		document.getElementById("rationale").innerText = rationaleText;

// // 		/* -------- EVIDENCE -------- */
// // 		let evidenceText = "Similar historical incidents:\n\n";
// // 		data.similar_incidents.forEach((item, idx) => {
// // 			evidenceText += `(${idx + 1}) [${item.incident_type}] `;
// // 			evidenceText += `similarity=${item.similarity.toFixed(3)}\n`;
// // 			evidenceText += `${item.text}\n\n`;
// // 		});
// // 		document.getElementById("evidence").innerText = evidenceText;
// // 	} catch (err) {
// // 		document.getElementById("situation").innerText =
// // 			"Error communicating with backend.";
// // 	}
// // }

// // async function exportReport() {
// // 	await fetch("http://127.0.0.1:8000/export", {
// // 		method: "POST",
// // 		headers: { "Content-Type": "application/json" },
// // 		body: JSON.stringify({
// // 			incident_text: document.getElementById("incidentInput").value,
// // 		}),
// // 	});

// // 	alert("Incident report generated.");
// // }

// // async function submitOverride() {
// // 	const note = document.getElementById("overrideNote").value;

// // 	await fetch("http://127.0.0.1:8000/override", {
// // 		method: "POST",
// // 		headers: { "Content-Type": "application/json" },
// // 		body: JSON.stringify({
// // 			corrected_incident_type: "Manual Review",
// // 			analyst_note: note,
// // 		}),
// // 	});

// // 	alert("Analyst override recorded.");
// // }


// function renderMarkdown(text) {
//   if (window.markdown && window.markdown.render) {
//     return window.markdown.render(text);
//   }
//   // Fallback: plain text
//   return `<pre>${text}</pre>`;
// }

// function openTab(tabId, evt) {
//   document
//     .querySelectorAll(".tab-content")
//     .forEach((t) => t.classList.remove("active"));
//   document
//     .querySelectorAll(".tab-btn")
//     .forEach((b) => b.classList.remove("active"));

//   document.getElementById(tabId).classList.add("active");
//   if (evt && evt.target) evt.target.classList.add("active");
// }

// async function analyze() {
//   const input = document.getElementById("incidentInput").value;
//   if (!input.trim()) {
//     alert("Please enter an incident description.");
//     return;
//   }

//   // ✅ FIX: Remove event parameter - it's not defined
//   openTab("situation");

//   document.getElementById("situation").innerText = "Analyzing incident...";
//   document.getElementById("plan").innerText = "";
//   document.getElementById("rationale").innerText = "";
//   document.getElementById("evidence").innerText = "";

//   try {
//     const data = await window.api.analyzeIncident(input);
    
//     // ✅ FIX: Add logging to see what data we receive
//     console.log("Received data from backend:", data);

//     /* -------- SITUATION -------- */
//     document.getElementById("situation").innerText =
//       `Severity: ${data.severity.level} (Score: ${data.severity.score})\n\n` +
//       `Incident Type: ${data.incident_type}\n` +
//       `Classification Confidence: ${(
//         data.classification_confidence * 100
//       ).toFixed(2)}%\n\n` +
//       data.situation_assessment;

//     /* -------- RESPONSE PLAN -------- */
//     let planText = "";
//     data.actions.forEach((a, idx) => {
//       planText += `${idx + 1}. ${a.action_id} (${a.phase})\n`;
//       planText += `   Relative relevance score: ${a.confidence.toFixed(
//         1,
//       )}%\n\n`;
//     });
//     document.getElementById("plan").innerText = planText;

//     /* -------- RATIONALE -------- */
//     let rationaleText = "";

//     // Normalize explanations into a map
//     let explanationMap = {};

//     if (Array.isArray(data.explanations)) {
//       data.explanations.forEach((e) => {
//         explanationMap[e.action_id] = e.explanation;
//       });
//     } else if (typeof data.explanations === "string") {
//       try {
//         const parsed = JSON.parse(data.explanations);
//         parsed.forEach((e) => {
//           explanationMap[e.action_id] = e.explanation;
//         });
//       } catch (err) {
//         console.error("Failed to parse explanations:", err);
//       }
//     }

//     data.actions.forEach((action) => {
//       rationaleText += "----------------------------------------\n";
//       rationaleText += `Action: ${action.action_id}\n\n`;
//       rationaleText += explanationMap[action.action_id]
//         ? explanationMap[action.action_id]
//         : "No explanation available.\n";
//       rationaleText += "\n\n";
//     });

//     document.getElementById("rationale").innerText = rationaleText;

//     /* -------- EVIDENCE -------- */
//     let evidenceText = "Similar historical incidents:\n\n";
//     data.similar_incidents.forEach((item, idx) => {
//       evidenceText += `(${idx + 1}) [${item.incident_type}] `;
//       evidenceText += `similarity=${item.similarity.toFixed(3)}\n`;
//       evidenceText += `${item.text}\n\n`;
//     });
//     document.getElementById("evidence").innerText = evidenceText;
    
//   } catch (err) {
//     // ✅ FIX: Add proper error logging and display
//     console.error("Backend communication error:", err);
//     document.getElementById("situation").innerText =
//       `Error communicating with backend:\n${err.message}\n\nCheck console for details.`;
//   }
// }

// async function exportReport() {
//   try {
//     const response = await fetch("http://127.0.0.1:8000/export", {
//       method: "POST",
//       headers: { "Content-Type": "application/json" },
//       body: JSON.stringify({
//         incident_text: document.getElementById("incidentInput").value,
//       }),
//     });

//     // ✅ FIX: Check if request was successful
//     if (!response.ok) {
//       throw new Error(`Export failed: ${response.status}`);
//     }

//     alert("Incident report generated.");
//   } catch (err) {
//     // ✅ FIX: Handle errors
//     console.error("Export error:", err);
//     alert(`Failed to export report: ${err.message}`);
//   }
// }

// async function submitOverride() {
//   const note = document.getElementById("overrideNote").value;

//   try {
//     const response = await fetch("http://127.0.0.1:8000/override", {
//       method: "POST",
//       headers: { "Content-Type": "application/json" },
//       body: JSON.stringify({
//         corrected_incident_type: "Manual Review",
//         analyst_note: note,
//       }),
//     });

//     // ✅ FIX: Check if request was successful
//     if (!response.ok) {
//       throw new Error(`Override failed: ${response.status}`);
//     }

//     alert("Analyst override recorded.");
//   } catch (err) {
//     // ✅ FIX: Handle errors
//     console.error("Override error:", err);
//     alert(`Failed to submit override: ${err.message}`);
//   }
// }

// ✅ Helper function to render markdown
function renderMarkdown(text) {
  if (window.marked) {
    return marked.parse(text);
  }
  // Fallback if marked isn't loaded
  return text.replace(/\n/g, '<br>');
}

function openTab(tabId, evt) {
  document
    .querySelectorAll(".tab-content")
    .forEach((t) => t.classList.remove("active"));
  document
    .querySelectorAll(".tab-btn")
    .forEach((b) => b.classList.remove("active"));

  document.getElementById(tabId).classList.add("active");
  if (evt && evt.target) evt.target.classList.add("active");
}

async function analyze() {
  const input = document.getElementById("incidentInput").value;
  if (!input.trim()) {
    alert("Please enter an incident description.");
    return;
  }

  openTab("situation");

  // Show loading message
  const situationEl = document.getElementById("situation");
  situationEl.innerHTML = "<p>Analyzing incident...</p>";
  situationEl.classList.add("markdown-content");
  
  document.getElementById("planContent").innerText = "";
  document.getElementById("rationale").innerText = "";
  document.getElementById("evidence").innerText = "";

  try {
    const data = await window.api.analyzeIncident(input);
    
    console.log("Received data from backend:", data);

    /* -------- SITUATION -------- */
    // Build markdown content
    let situationMarkdown = `## Incident Analysis\n\n`;
    situationMarkdown += `**Severity:** ${data.severity.level} (Score: ${data.severity.score})\n\n`;
    situationMarkdown += `**Incident Type:** ${data.incident_type}\n\n`;
    situationMarkdown += `**Classification Confidence:** ${(data.classification_confidence * 100).toFixed(2)}%\n\n`;
    situationMarkdown += `---\n\n`;
    situationMarkdown += `### Situation Assessment\n\n`;
    situationMarkdown += data.situation_assessment;

    // ✅ Render markdown and set innerHTML
    situationEl.innerHTML = renderMarkdown(situationMarkdown);
    situationEl.classList.add("markdown-content");

    /* -------- RESPONSE PLAN -------- */
    let planText = "";
    data.actions.forEach((a, idx) => {
      planText += `${idx + 1}. ${a.action_id} (${a.phase})\n`;
      planText += `   Relative relevance score: ${a.confidence.toFixed(1)}%\n\n`;
    });
    document.getElementById("planContent").innerText = planText;

    /* -------- RATIONALE -------- */
    let rationaleText = "";

    // Normalize explanations into a map
    let explanationMap = {};

    if (Array.isArray(data.explanations)) {
      data.explanations.forEach((e) => {
        explanationMap[e.action_id] = e.explanation;
      });
    } else if (typeof data.explanations === "string") {
      try {
        const parsed = JSON.parse(data.explanations);
        parsed.forEach((e) => {
          explanationMap[e.action_id] = e.explanation;
        });
      } catch (err) {
        console.error("Failed to parse explanations:", err);
      }
    }

    // ✅ Build markdown for rationale
    let rationaleMarkdown = "# Action Rationale\n\n";
    data.actions.forEach((action) => {
      rationaleMarkdown += `---\n\n`;
      rationaleMarkdown += `## ${action.action_id}\n\n`;
      rationaleMarkdown += explanationMap[action.action_id]
        ? explanationMap[action.action_id] + "\n\n"
        : "*No explanation available.*\n\n";
    });

    const rationaleEl = document.getElementById("rationale");
    rationaleEl.innerHTML = renderMarkdown(rationaleMarkdown);
    rationaleEl.classList.add("markdown-content");

    /* -------- EVIDENCE -------- */
    let evidenceText = "Similar historical incidents:\n\n";
    data.similar_incidents.forEach((item, idx) => {
      evidenceText += `(${idx + 1}) [${item.incident_type}] `;
      evidenceText += `similarity=${item.similarity.toFixed(3)}\n`;
      evidenceText += `${item.text}\n\n`;
    });
    document.getElementById("evidence").innerText = evidenceText;
    
  } catch (err) {
    console.error("Backend communication error:", err);
    situationEl.innerHTML =
      `<p style="color: #ef4444;"><strong>Error communicating with backend:</strong><br>${err.message}</p>` +
      `<p>Check console for details.</p>`;
    situationEl.classList.add("markdown-content");
  }
}

async function exportReport() {
  try {
    const response = await fetch("http://127.0.0.1:8000/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        incident_text: document.getElementById("incidentInput").value,
      }),
    });

    if (!response.ok) {
      throw new Error(`Export failed: ${response.status}`);
    }

    alert("Incident report generated.");
  } catch (err) {
    console.error("Export error:", err);
    alert(`Failed to export report: ${err.message}`);
  }
}

async function submitOverride() {
  const note = document.getElementById("overrideNote").value;

  try {
    const response = await fetch("http://127.0.0.1:8000/override", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        corrected_incident_type: "Manual Review",
        analyst_note: note,
      }),
    });

    if (!response.ok) {
      throw new Error(`Override failed: ${response.status}`);
    }

    alert("Analyst override recorded.");
  } catch (err) {
    console.error("Override error:", err);
    alert(`Failed to submit override: ${err.message}`);
  }
}