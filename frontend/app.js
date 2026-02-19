/**
 * app.js — Frontend logic for Meeting Minutes Generator
 *
 * Handles:
 * - File drag & drop + selection
 * - API calls to FastAPI backend
 * - UI state management
 * - Copy/download functionality
 * - Q&A interactions
 */

// ── DOM References ─────────────────────────────────────────────────────────────
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const fileInfo = document.getElementById("fileInfo");
const fileName = document.getElementById("fileName");
const fileSize = document.getElementById("fileSize");
const clearFile = document.getElementById("clearFile");
const generateBtn = document.getElementById("generateBtn");
const processingStatus = document.getElementById("processingStatus");
const metadata = document.getElementById("metadata");
const qaSection = document.getElementById("qaSection");
const questionInput = document.getElementById("questionInput");
const askBtn = document.getElementById("askBtn");
const qaAnswer = document.getElementById("qaAnswer");
const qaAnswerText = document.getElementById("qaAnswerText");
const emptyState = document.getElementById("emptyState");
const errorState = document.getElementById("errorState");
const errorMessage = document.getElementById("errorMessage");
const outputContent = document.getElementById("outputContent");
const minutesText = document.getElementById("minutesText");
const outputActions = document.getElementById("outputActions");
const copyBtn = document.getElementById("copyBtn");
const downloadBtn = document.getElementById("downloadBtn");
const statusBadge = document.getElementById("statusBadge");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");

// API Base URL — points to our FastAPI backend
const API_BASE = "http://localhost:8000/api";

// State
let selectedFile = null;
let currentMinutes = "";
let processingSteps = ["step1", "step2", "step3", "step4", "step5"];
let stepTimers = [];

// ── Startup: Check Server Status ─────────────────────────────────────────────
async function checkServerStatus() {
  try {
    const response = await fetch(`${API_BASE}/status`);
    const data = await response.json();

    if (data.status === "ready") {
      statusDot.classList.add("ready");
      statusText.textContent = "Pipeline Ready";
    } else {
      statusText.textContent = "Loading Models...";
      // Check again in 5 seconds
      setTimeout(checkServerStatus, 5000);
    }
  } catch (err) {
    statusDot.classList.add("error");
    statusText.textContent = "Server Offline";
    setTimeout(checkServerStatus, 5000);
  }
}

// ── File Handling ─────────────────────────────────────────────────────────────
dropzone.addEventListener("click", () => fileInput.click());

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
  if (file) handleFileSelection(file);
});

fileInput.addEventListener("change", (e) => {
  if (e.target.files[0]) handleFileSelection(e.target.files[0]);
});

clearFile.addEventListener("click", resetFileSelection);

function handleFileSelection(file) {
  // Validate file type
  if (!file.name.match(/\.(txt|md)$/i)) {
    showToast("Please select a .txt or .md file", "error");
    return;
  }

  // Validate file size (5MB max)
  if (file.size > 5 * 1024 * 1024) {
    showToast("File too large. Maximum size is 5MB", "error");
    return;
  }

  selectedFile = file;

  // Show file info
  fileName.textContent = file.name;
  fileSize.textContent = formatFileSize(file.size);
  fileInfo.style.display = "flex";
  dropzone.style.display = "none";
  generateBtn.disabled = false;
}

function resetFileSelection() {
  selectedFile = null;
  fileInput.value = "";
  fileInfo.style.display = "none";
  dropzone.style.display = "block";
  generateBtn.disabled = true;
}

function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

// ── Generate Minutes ──────────────────────────────────────────────────────────
generateBtn.addEventListener("click", generateMinutes);

async function generateMinutes() {
  if (!selectedFile) return;

  // Show processing UI
  showProcessingUI();
  startStepAnimation();

  try {
    // Create FormData to send file
    const formData = new FormData();
    formData.append("file", selectedFile);

    // Call the API
    const response = await fetch(`${API_BASE}/upload-transcript`, {
      method: "POST",
      body: formData,
      // Note: Don't set Content-Type header — the browser sets it automatically
      // with the correct multipart boundary for FormData
    });

    // Stop step animation
    stopStepAnimation();

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(
        errorData.detail ||
          `HTTP ${response.status}: Failed to generate minutes`,
      );
    }

    const data = await response.json();

    // Display the results
    displayResults(data);
  } catch (err) {
    stopStepAnimation();
    showError(err.message);
  }
}

function showProcessingUI() {
  generateBtn.disabled = true;
  generateBtn.textContent = "⏳ Processing...";
  generateBtn.classList.add("loading");

  processingStatus.style.display = "block";
  metadata.style.display = "none";
  emptyState.style.display = "none";
  errorState.style.display = "none";
  outputContent.style.display = "none";
  outputActions.style.display = "none";

  // Reset all steps to pending
  processingSteps.forEach((stepId) => {
    const el = document.getElementById(stepId);
    el.classList.remove("active", "done");
    el.querySelector(".step-icon").textContent = "⏳";
  });
}

function startStepAnimation() {
  // Simulate progress through steps
  // Each step activates after a delay to show progress
  const delays = [0, 3000, 6000, 9000, 12000];

  processingSteps.forEach((stepId, i) => {
    const timer = setTimeout(() => {
      // Mark previous step as done
      if (i > 0) {
        const prevStep = document.getElementById(processingSteps[i - 1]);
        prevStep.classList.remove("active");
        prevStep.classList.add("done");
        prevStep.querySelector(".step-icon").textContent = "✅";
      }
      // Activate current step
      const currentStep = document.getElementById(stepId);
      currentStep.classList.add("active");
      currentStep.querySelector(".step-icon").textContent = "🔄";
    }, delays[i]);

    stepTimers.push(timer);
  });
}

function stopStepAnimation() {
  stepTimers.forEach((t) => clearTimeout(t));
  stepTimers = [];
}

function displayResults(data) {
  currentMinutes = data.minutes;

  // Show output
  minutesText.textContent = data.minutes;
  outputContent.style.display = "block";
  outputActions.style.display = "flex";
  processingStatus.style.display = "none";
  emptyState.style.display = "none";

  // Update metadata
  document.getElementById("metaChunks").textContent = data.chunks_created;
  document.getElementById("metaSections").textContent = data.sections_retrieved;
  document.getElementById("metaTime").textContent =
    `${data.processing_time_seconds}s`;
  metadata.style.display = "block";

  // Show Q&A section
  qaSection.style.display = "block";
  qaAnswer.style.display = "none";

  // Reset button
  generateBtn.disabled = false;
  generateBtn.innerHTML =
    '<span class="btn-icon">⚡</span> Generate Meeting Minutes';
  generateBtn.classList.remove("loading");
}

function showError(message) {
  errorMessage.textContent = message;
  errorState.style.display = "flex";
  emptyState.style.display = "none";
  processingStatus.style.display = "none";

  generateBtn.disabled = false;
  generateBtn.innerHTML =
    '<span class="btn-icon">⚡</span> Generate Meeting Minutes';
  generateBtn.classList.remove("loading");
}

// ── Q&A ───────────────────────────────────────────────────────────────────────
askBtn.addEventListener("click", askQuestion);
questionInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter") askQuestion();
});

async function askQuestion() {
  const question = questionInput.value.trim();
  if (!question) return;

  askBtn.textContent = "...";
  askBtn.disabled = true;

  try {
    const response = await fetch(`${API_BASE}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    if (!response.ok) throw new Error("Failed to get answer");

    const data = await response.json();
    qaAnswerText.textContent = data.answer;
    qaAnswer.style.display = "block";
  } catch (err) {
    qaAnswerText.textContent = "Error: " + err.message;
    qaAnswer.style.display = "block";
  } finally {
    askBtn.textContent = "Ask";
    askBtn.disabled = false;
  }
}

// ── Copy & Download ───────────────────────────────────────────────────────────
copyBtn.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(currentMinutes);
    copyBtn.textContent = "✅ Copied!";
    setTimeout(() => (copyBtn.textContent = "📋 Copy"), 2000);
  } catch (err) {
    // Fallback for older browsers
    const textArea = document.createElement("textarea");
    textArea.value = currentMinutes;
    document.body.appendChild(textArea);
    textArea.select();
    document.execCommand("copy");
    document.body.removeChild(textArea);
    copyBtn.textContent = "✅ Copied!";
    setTimeout(() => (copyBtn.textContent = "📋 Copy"), 2000);
  }
});

downloadBtn.addEventListener("click", () => {
  const blob = new Blob([currentMinutes], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `meeting-minutes-${new Date().toISOString().split("T")[0]}.txt`;
  a.click();
  URL.revokeObjectURL(url);
});

// ── Utilities ─────────────────────────────────────────────────────────────────
function resetUI() {
  errorState.style.display = "none";
  emptyState.style.display = "flex";
  outputContent.style.display = "none";
  outputActions.style.display = "none";
}

function showToast(message, type = "info") {
  // Simple toast notification
  const toast = document.createElement("div");
  toast.style.cssText = `
        position: fixed; bottom: 2rem; right: 2rem;
        background: ${type === "error" ? "#ef4444" : "#4361ee"};
        color: white; padding: 0.75rem 1.25rem;
        border-radius: 8px; font-size: 0.85rem;
        z-index: 1000; box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        animation: slideIn 0.3s ease;
    `;
  toast.textContent = message;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3000);
}

// ── Init ──────────────────────────────────────────────────────────────────────
checkServerStatus();
