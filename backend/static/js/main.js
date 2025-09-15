// Mic recording
let micRecorder, micChunks = [];
const micStart = document.getElementById("micStart");
const micStop = document.getElementById("micStop");
const micStatus = document.getElementById("micStatus");

const getLanguage = () => {
  const sel = document.getElementById("languageSelect");
  return sel ? sel.value : "auto";
};

if (micStart) {
  micStart.addEventListener("click", async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      micRecorder = new MediaRecorder(stream);
      micChunks = [];

      micRecorder.ondataavailable = e => { if (e.data.size) micChunks.push(e.data); };
      micRecorder.onstop = async () => {
        const blob = new Blob(micChunks, { type: "audio/webm" });
        const fd = new FormData();
        fd.append("audio", blob, "mic_recording.webm");
        fd.append("language", getLanguage());

        micStatus.textContent = "Uploading & processing...";
        const res = await fetch("/record", { method: "POST", body: fd });
        const data = await res.json();
        if (data.redirect_url) {
          window.location.href = data.redirect_url;
        } else {
          micStatus.textContent = "Error: " + (data.error || "unknown");
        }
      };

      micRecorder.start();
      micStatus.textContent = "Recording (mic)...";
      micStart.classList.add("hidden");
      micStop.classList.remove("hidden");
    } catch (err) {
      micStatus.textContent = "Permission denied or error: " + err.message;
    }
  });

  micStop.addEventListener("click", () => {
    if (micRecorder && micRecorder.state !== "inactive") micRecorder.stop();
    micStatus.textContent = "Processing...";
    micStart.classList.remove("hidden");
    micStop.classList.add("hidden");
  });
}

// Tab/System recording
let tabRecorder, tabChunks = [];
const tabStart = document.getElementById("tabStart");
const tabStop = document.getElementById("tabStop");
const tabStatus = document.getElementById("tabStatus");

if (tabStart) {
  tabStart.addEventListener("click", async () => {
    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({ audio: true, video: true });

      tabRecorder = new MediaRecorder(stream);
      tabChunks = [];

      tabRecorder.ondataavailable = e => { if (e.data.size) tabChunks.push(e.data); };
      tabRecorder.onstop = async () => {
        const blob = new Blob(tabChunks, { type: "audio/webm" });
        const fd = new FormData();
        fd.append("audio", blob, "tab_recording.webm");
        fd.append("language", getLanguage());

        tabStatus.textContent = "Uploading & processing...";
        const res = await fetch("/record", { method: "POST", body: fd });
        const data = await res.json();
        if (data.redirect_url) {
          window.location.href = data.redirect_url;
        } else {
          tabStatus.textContent = "Error: " + (data.error || "unknown");
        }
      };

      tabRecorder.start();
      tabStatus.textContent = "Recording (tab/system)...";
      tabStart.classList.add("hidden");
      tabStop.classList.remove("hidden");
    } catch (err) {
      tabStatus.textContent = "Permission denied or error: " + err.message;
    }
  });

  tabStop.addEventListener("click", () => {
    if (tabRecorder && tabRecorder.state !== "inactive") tabRecorder.stop();
    tabStatus.textContent = "Processing...";
    tabStart.classList.remove("hidden");
    tabStop.classList.add("hidden");
  });
}
