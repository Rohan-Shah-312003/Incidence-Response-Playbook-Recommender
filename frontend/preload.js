const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("api", {
	analyzeIncident: async (text) => {
		const response = await fetch("http://127.0.0.1:8000/analyze", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ incident_text: text }),
		});

		if (!response.ok) {
			throw new Error("Backend error");
		}

		return response.json();
	},
});

contextBridge.exposeInMainWorld("electronAPI", {
	controlWindow: (action) => ipcRenderer.send("window-controls", action),
});

contextBridge.exposeInMainWorld("env", {
	platform: process.platform,
});
