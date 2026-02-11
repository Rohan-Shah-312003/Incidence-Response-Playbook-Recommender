const { app, BrowserWindow, nativeImage } = require("electron");
const path = require("path");

function createWindow() {
	const win = new BrowserWindow({
		width: 1000,
		height: 700,
		titleBarStyle: "hidden",
		trafficLightPosition: { x: 18, y: 13 },
		icon: process.platform === 'darwin' 
            ? path.join(__dirname, "logo.icns")
            : path.join(__dirname, "logo.ico"),
		webPreferences: {
			preload: path.join(__dirname, "preload.js"),
			contextIsolation: true,
			nodeIntegration: false,
		},
	});
	win.loadFile("index.html");

	// win.webContents.openDevTools(); // open dev tools
}

// app.whenReady().then(() => {
//     // Set dock icon for macOS (development mode)
//     if (process.platform === 'darwin') {
//         const icon = nativeImage.createFromPath(path.join(__dirname, 'logo.icns'));
//         app.dock.setIcon(icon);
//     }
    
//     createWindow();
// });


app.whenReady().then(createWindow);

app.on("window-all-closed", () => {
    if (process.platform !== "darwin") app.quit();
});

app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
});