/**
 * Frontend Application Script: app.js
 * Decoupled Architecture for Infant Safety & Cry Monitoring
 * Zero Emojis / Zero Icons Policy
 * Ready for Remote API Server (configure API_BASE_URL)
 */

// Base URL for API calls. Empty string uses current host (http://127.0.0.1:8000).
// In future, set this to the remote cloud server endpoint, e.g. "https://api.synexan.com"
const API_BASE_URL = window.location.origin;

// DOM Elements
const inputVideo = document.getElementById("inputVideo");
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");

const outputFrame = document.getElementById("outputFrame");
const resultPlaceholder = document.getElementById("resultPlaceholder");

const runBtn = document.getElementById("runBtn");
const stopBtn = document.getElementById("stopBtn");
const statusBanner = document.getElementById("statusBanner");

const visionStatus = document.getElementById("visionStatus");
const audioStatus = document.getElementById("audioStatus");
const causeStatus = document.getElementById("causeStatus");
const examplesGrid = document.getElementById("examplesGrid");

// State
let currentVideoPath = "";
let eventSource = null;
let isAnalyzing = false;

// 1. Initial Load: Fetch Test Examples with Visual Thumbnails
async function loadExamples() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/examples`);
        if (!response.ok) return;
        const data = await response.json();
        const examples = data.examples || [];

        examplesGrid.innerHTML = "";
        examples.forEach((ex, idx) => {
            const card = document.createElement("div");
            card.className = "example-card";
            card.innerHTML = `
                <div class="example-thumb-wrapper">
                    <img class="example-thumb-img" src="${ex.thumbnail_url}" alt="${ex.filename}">
                </div>
                <div class="example-name">${ex.filename}</div>
            `;
            card.addEventListener("click", () => {
                selectExample(ex.path, card, ex.thumbnail_url);
            });
            examplesGrid.appendChild(card);

            // Auto-select first example video
            if (idx === 0 && !currentVideoPath) {
                selectExample(ex.path, card, ex.thumbnail_url);
            }
        });
    } catch (err) {
        console.error("Error loading examples:", err);
    }
}

// 2. Select an Example Video
function selectExample(path, cardElement, thumbnailUrl) {
    if (isAnalyzing) {
        stopAnalysis();
    }

    currentVideoPath = path;

    // Highlight selected card
    document.querySelectorAll(".example-card").forEach(c => c.classList.remove("selected"));
    if (cardElement) {
        cardElement.classList.add("selected");
    }

    // Set video src and poster for visual frame 1
    inputVideo.src = `${API_BASE_URL}/api/video?path=${encodeURIComponent(path)}`;
    if (thumbnailUrl) {
        inputVideo.poster = thumbnailUrl;
    }
    inputVideo.style.display = "block";
    dropzone.style.display = "none";
    inputVideo.load();

    resetOutputs();
}

// 3. File Upload Handling
dropzone.addEventListener("click", () => {
    fileInput.click();
});

fileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) {
        handleUploadFile(file);
    }
});

dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
});

dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file) {
        handleUploadFile(file);
    }
});

async function handleUploadFile(file) {
    if (isAnalyzing) {
        stopAnalysis();
    }

    // Unselect examples
    document.querySelectorAll(".examples-table tbody tr").forEach(r => r.classList.remove("selected"));

    const formData = new FormData();
    formData.append("file", file);

    statusBanner.className = "status-banner";
    statusBanner.style.backgroundColor = "#F1F5F9";
    statusBanner.style.color = "#475569";
    statusBanner.textContent = `DANG TAI LEN: ${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)...`;

    try {
        const response = await fetch(`${API_BASE_URL}/api/upload`, {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error("Loi khi tai len video.");
        }

        const data = await response.json();
        currentVideoPath = data.path;

        inputVideo.src = `${API_BASE_URL}/api/video?path=${encodeURIComponent(currentVideoPath)}`;
        inputVideo.style.display = "block";
        dropzone.style.display = "none";
        inputVideo.load();

        statusBanner.textContent = `SAN SANG: Da tai len thanh cong ${data.filename}. Vui long bam "Phan Tich Video".`;
        resetOutputs();
    } catch (err) {
        console.error("Upload failed:", err);
        statusBanner.textContent = `LOI: Khong the tai len video. Vui long thu lai.`;
    }
}

// 4. Reset Outputs
function resetOutputs() {
    outputFrame.style.display = "none";
    outputFrame.src = "";
    resultPlaceholder.style.display = "block";

    visionStatus.textContent = "-";
    audioStatus.textContent = "-";
    causeStatus.textContent = "-";
}

// 5. Start Streaming Analysis
function startAnalysis() {
    if (!currentVideoPath) {
        alert("Vui long chon video mau hoac tai len video truoc khi phan tich.");
        return;
    }

    if (isAnalyzing) return;

    isAnalyzing = true;
    runBtn.disabled = true;
    stopBtn.disabled = false;

    resultPlaceholder.style.display = "none";
    outputFrame.style.display = "block";

    // Play input video in sync
    inputVideo.currentTime = 0;
    inputVideo.play().catch(() => {});

    const streamUrl = `${API_BASE_URL}/api/analyze/stream?video_path=${encodeURIComponent(currentVideoPath)}`;
    eventSource = new EventSource(streamUrl);

    eventSource.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);

            if (data.event === "done") {
                finishAnalysis();
                return;
            }

            if (data.event === "step") {
                // Update live video frame with annotations
                outputFrame.src = `data:image/jpeg;base64,${data.frame_b64}`;

                // Update Safety Status Banner
                statusBanner.className = "status-banner active";
                statusBanner.style.backgroundColor = data.color || "#059669";
                statusBanner.style.color = "#FFFFFF";
                statusBanner.innerHTML = `
                    <div class="banner-title">${data.title}</div>
                    <div class="banner-desc">${data.description}</div>
                `;

                // Update Metric KPI Cards
                visionStatus.textContent = data.vision_text;
                audioStatus.textContent = data.audio_text;
                causeStatus.textContent = data.cause_text;
            }
        } catch (err) {
            console.error("Error processing SSE message:", err);
        }
    };

    eventSource.onerror = (err) => {
        console.warn("SSE connection closed or error occurred:", err);
        finishAnalysis();
    };
}

// 7. Stop Analysis
function stopAnalysis() {
    if (eventSource) {
        eventSource.close();
        eventSource = null;
    }
    inputVideo.pause();
    isAnalyzing = false;
    runBtn.disabled = false;
    stopBtn.disabled = true;
}

function finishAnalysis() {
    stopAnalysis();
    console.log("Analysis stream completed.");
}

// Event Bindings
runBtn.addEventListener("click", startAnalysis);
stopBtn.addEventListener("click", stopAnalysis);

// Initialize on DOM Ready
document.addEventListener("DOMContentLoaded", () => {
    loadExamples();
});
