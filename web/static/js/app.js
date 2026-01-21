// Violence Detection Web App - Main JavaScript

// ==================== INITIALIZATION ====================

// API base URL
const API_BASE = window.location.origin;

// Socket.IO connection
const socket = io(API_BASE);

// State
let webcamStream = null;
let webcamInterval = null;
let isWebcamActive = false;
let charts = {}; // Store chart instances
let frameSkipCounter = 0; // Skip frames for better performance

// ==================== DOM ELEMENTS & EVENT LISTENERS ====================

document.addEventListener('DOMContentLoaded', () => {
    console.log('App initialized');
    checkAPIStatus();
    
    // Initialize Navigation Tabs
    initTabs();

    // Initialize specific page components
    initUploadPage();
    initWebcamPage();
    initHistoryPage();
    initNotifications();
    
    // Default to Upload tab or active tab
    const activeTab = document.querySelector('.tab.active');
    if (activeTab) {
        switchTab(activeTab.dataset.tab);
    }
});

// Tab Switching Logic
function initTabs() {
    const tabButtons = document.querySelectorAll('.tab');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabName = button.dataset.tab;
            switchTab(tabName);
        });
    });
}

function switchTab(tabName) {
    // Update buttons
    const tabButtons = document.querySelectorAll('.tab');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabButtons.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabName);
    });
    
    // Update contents
    tabContents.forEach(content => {
        content.classList.toggle('active', content.id === `${tabName}-tab`);
    });
    
    // Load specific data if needed
    if (tabName === 'history') {
        loadHistory();
        loadStatistics();
    }
}

// Notifications
function initNotifications() {
    // Override socket error handler
    socket.on('error', (data) => {
        console.error('Socket error:', data);
        showNotification(data.message || 'Có lỗi xảy ra', 'error');
    });
    
    socket.on('connect', () => {
        console.log('Connected to server');
    });
}

// ==================== PAGE SPECIFIC LOGIC ====================

// --- UPLOAD PAGE ---
function initUploadPage() {
    const uploadArea = document.getElementById('upload-area');
    const videoInput = document.getElementById('video-input');
    
    if (uploadArea && videoInput) {
        uploadArea.addEventListener('click', () => videoInput.click());
        uploadArea.addEventListener('dragover', handleDragOver);
        uploadArea.addEventListener('dragleave', handleDragLeave);
        uploadArea.addEventListener('drop', handleDrop);
        videoInput.addEventListener('change', handleFileSelect);
    }
}

// --- WEBCAM PAGE ---
function initWebcamPage() {
    const startWebcamBtn = document.getElementById('start-webcam-btn');
    const stopWebcamBtn = document.getElementById('stop-webcam-btn');
    
    if (startWebcamBtn && stopWebcamBtn) {
        startWebcamBtn.addEventListener('click', startWebcam);
        stopWebcamBtn.addEventListener('click', stopWebcam);
        
        // Socket events for webcam
        socket.on('prediction', (result) => {
            displayWebcamResult(result);
        });
    }
}

// --- HISTORY PAGE ---
function initHistoryPage() {
    const historyFilter = document.getElementById('history-filter');
    const refreshHistoryBtn = document.getElementById('refresh-history-btn');
    const dateRangePicker = document.getElementById('date-range-picker');
    
    if (historyFilter || refreshHistoryBtn) {
        if (historyFilter) historyFilter.addEventListener('change', loadHistory);
        if (refreshHistoryBtn) refreshHistoryBtn.addEventListener('click', () => {
            loadHistory();
            loadStatistics();
        });
        
        // Initialize Date Picker
        if (dateRangePicker && window.flatpickr) {
            flatpickr(dateRangePicker, {
                mode: "range",
                dateFormat: "Y-m-d",
                onChange: function(selectedDates, dateStr) {
                    loadHistory(); 
                }
            });
        }
    }
}


// ==================== SHARED FUNCTIONS ====================

function showNotification(message, type = 'info') {
    if (typeof Toastify === 'function') {
        Toastify({
            text: message,
            duration: 3000,
            gravity: 'bottom',
            position: 'right',
            className: `toast-${type}`,
            stopOnFocus: true
        }).showToast();
    } else {
        console.log(`[${type.toUpperCase()}] ${message}`);
    }
}

async function checkAPIStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/status`);
        const data = await response.json();
        if (!data.ml_ready) {
             console.warn('Model not ready');
        }
    } catch (error) {
        console.error('API Status Error:', error);
    }
}

// ==================== UPLOAD HANDLERS ====================

function handleDragOver(e) {
    e.preventDefault();
    document.getElementById('upload-area').classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    document.getElementById('upload-area').classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    document.getElementById('upload-area').classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
}

function handleFileSelect(e) {
    if (e.target.files.length > 0) handleFile(e.target.files[0]);
}

async function handleFile(file) {
    const uploadProgress = document.getElementById('upload-progress');
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');
    const uploadResult = document.getElementById('upload-result');
    const videoPreview = document.getElementById('video-preview');
    
    // Validation
    const validTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska'];
    if (!validTypes.includes(file.type) && !file.name.match(/\.(mp4|avi|mov|mkv)$/i)) {
        showNotification('Định dạng file không hợp lệ!', 'error');
        return;
    }
    
    if (file.size > 100 * 1024 * 1024) {
        showNotification('File quá lớn! Tối đa 100MB', 'error');
        return;
    }

    // UI Reset
    uploadProgress.classList.remove('hidden');
    uploadResult.classList.add('hidden');
    videoPreview.classList.add('hidden');
    progressFill.style.width = '0%';
    progressText.textContent = 'Đang upload...';

    try {
        const formData = new FormData();
        formData.append('video', file);
        
        const uploadResponse = await fetch(`${API_BASE}/api/upload`, {
            method: 'POST',
            body: formData
        });
        
        const uploadData = await uploadResponse.json();
        if (!uploadData.success) throw new Error(uploadData.error || 'Upload failed');
        
        progressFill.style.width = '50%';
        progressText.textContent = 'Đang phân tích (có thể mất vài phút)...';
        
        const predictResponse = await fetch(`${API_BASE}/api/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename: uploadData.filename })
        });
        
        const predictData = await predictResponse.json();
        if (!predictData.success) throw new Error(predictData.error || 'Prediction failed');
        
        progressFill.style.width = '100%';
        progressText.textContent = 'Hoàn thành!';
        
        setTimeout(() => {
            uploadProgress.classList.add('hidden');
            displayUploadResult(predictData.result, uploadData.filename);
        }, 500);
        
    } catch (error) {
        console.error('Error:', error);
        showNotification(error.message, 'error');
        uploadProgress.classList.add('hidden');
    }
}

function displayUploadResult(result, filename) {
    const resultContent = document.getElementById('result-content');
    const uploadResult = document.getElementById('upload-result');
    const videoPreview = document.getElementById('video-preview');
    const uploadedVideo = document.getElementById('uploaded-video');
    const videoSource = document.getElementById('video-source');
    const isViolent = result.is_violent;
    const className = isViolent ? 'violent' : 'safe';
    
    // Show video preview
    if (filename) {
        const videoUrl = `${API_BASE}/uploads/${filename}`;
        videoSource.src = videoUrl;
        uploadedVideo.load();
        videoPreview.classList.remove('hidden');
    }
    
    resultContent.innerHTML = `
        <div class="result-item ${className}">
            <span class="result-label">Kết quả:</span>
            <span class="result-value ${className}">${result.prediction}</span>
        </div>
        <div class="result-item">
            <span class="result-label">Độ tin cậy:</span>
            <span class="result-value">${result.confidence_percent}</span>
        </div>
        <div style="margin-top: 15px; text-align: center;">
            ${isViolent ? 
                '<div style="color: var(--danger); font-weight: bold; font-size: 1.1rem;">⚠️ CẢNH BÁO: PHÁT HIỆN BẠO LỰC</div>' : 
                '<div style="color: var(--success); font-weight: bold; font-size: 1.1rem;">✅ VIDEO AN TOÀN</div>'}
        </div>
    `;
    
    uploadResult.classList.remove('hidden');
}


// ==================== WEBCAM HANDLERS ====================

async function startWebcam() {
    try {
        const webcamVideo = document.getElementById('webcam-video');
        
        // Request HD quality with fallback
        const constraints = {
            video: {
                width: { exact: 1280 },
                height: { exact: 720 },
                frameRate: { ideal: 30, max: 30 },
                facingMode: 'user'
            }
        };
        
        try {
            webcamStream = await navigator.mediaDevices.getUserMedia(constraints);
        } catch (e) {
            // Fallback to ideal if exact fails
            console.warn('HD not available, falling back to ideal:', e);
            const fallbackConstraints = {
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    frameRate: { ideal: 30 },
                    facingMode: 'user'
                }
            };
            webcamStream = await navigator.mediaDevices.getUserMedia(fallbackConstraints);
        }
        
        webcamVideo.srcObject = webcamStream;
        
        // Wait for video metadata to load to get actual resolution
        webcamVideo.onloadedmetadata = () => {
            console.log(`Webcam resolution: ${webcamVideo.videoWidth}x${webcamVideo.videoHeight}`);
            // Force play to ensure video renders properly
            webcamVideo.play().catch(e => console.error('Error playing video:', e));
        };
        
        isWebcamActive = true;
        
        document.getElementById('start-webcam-btn').classList.add('hidden');
        document.getElementById('stop-webcam-btn').classList.remove('hidden');
        document.getElementById('webcam-overlay').classList.remove('hidden');
        
        // Initialize overlay with buffering state (matches CLI)
        document.getElementById('overlay-status-text').textContent = 'Buffering...';
        document.getElementById('overlay-conf-text').textContent = '0.00%';
        document.getElementById('overlay-buffer-text').textContent = '0/16';
        
        socket.emit('start_webcam');
        webcamInterval = setInterval(sendWebcamFrame, 100); // 10 FPS
        
        showNotification('Webcam đã bắt đầu!', 'success');
    } catch (error) {
        console.error('Webcam Error:', error);
        showNotification('Không thể truy cập camera: ' + error.message, 'error');
    }
}

function stopWebcam() {
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
    if (webcamInterval) {
        clearInterval(webcamInterval);
        webcamInterval = null;
    }
    isWebcamActive = false;
    frameSkipCounter = 0; // Reset skip counter
    
    document.getElementById('start-webcam-btn').classList.remove('hidden');
    document.getElementById('stop-webcam-btn').classList.add('hidden');
    document.getElementById('webcam-overlay').classList.add('hidden');
    document.getElementById('webcam-result').classList.add('hidden');
    document.getElementById('webcam-video').srcObject = null;
    
    socket.emit('stop_webcam');
    showNotification('Webcam đã dừng', 'info');
}

function sendWebcamFrame() {
    if (!isWebcamActive) return;
    
    const webcamVideo = document.getElementById('webcam-video');
    const webcamCanvas = document.getElementById('webcam-canvas');
    const context = webcamCanvas.getContext('2d');
    
    if (webcamVideo.videoWidth === 0) return;
    
    // Skip frames like CLI (process every 2nd frame)
    frameSkipCounter++;
    if (frameSkipCounter % 2 !== 0) return;
    
    // Match canvas size exactly to video to prevent any scaling
    const videoWidth = webcamVideo.videoWidth;
    const videoHeight = webcamVideo.videoHeight;
    
    if (webcamCanvas.width !== videoWidth || webcamCanvas.height !== videoHeight) {
        webcamCanvas.width = videoWidth;
        webcamCanvas.height = videoHeight;
    }
    
    // Draw at exact size - no scaling
    context.drawImage(webcamVideo, 0, 0, videoWidth, videoHeight);
    
    // Use near-lossless JPEG quality (0.98) 
    const frameData = webcamCanvas.toDataURL('image/jpeg', 0.98);
    socket.emit('webcam_frame', { frame: frameData });
}


function displayWebcamResult(result) {
    if (!isWebcamActive) return;
    
    // Elements
    const overlay = document.getElementById('webcam-overlay');
    const statusText = document.getElementById('overlay-status-text');
    const confText = document.getElementById('overlay-conf-text');
    const bufferText = document.getElementById('overlay-buffer-text');
    const statsBox = document.querySelector('.overlay-stats');
    const warningBox = document.getElementById('overlay-warning');
    
    const webcamResultContent = document.getElementById('webcam-result-content');
    const webcamResult = document.getElementById('webcam-result');

    const isViolent = result.is_violent;
    const prediction = result.prediction; // 'Violent' or 'Non-Violent'
    const confidence = parseFloat(result.confidence);
    const bufferSize = result.buffer_size || 16;
    
    // Update Stats text
    statusText.textContent = prediction;
    confText.textContent = (confidence * 100).toFixed(2) + '%';
    bufferText.textContent = `${bufferSize}/16`;

    // Styling based on result (Like CLI script)
    if (isViolent) {
        statusText.style.color = '#ef4444'; // Red
        statsBox.style.borderLeftColor = '#ef4444';
        
        // Show warning if violent & high confidence (matches CLI: confidence > 0.7)
        if (confidence > 0.7) {
            warningBox.classList.remove('hidden');
        } else {
            warningBox.classList.add('hidden');
        }
    } else {
        statusText.style.color = '#10b981'; // Green
        statsBox.style.borderLeftColor = '#10b981';
        warningBox.classList.add('hidden');
    }
    
    // Legacy result box update (keeps bottom card updated)
    webcamResultContent.innerHTML = `
        <div class="result-item ${isViolent ? 'violent' : 'safe'}">
            <span class="result-label">Dự đoán:</span>
            <span class="result-value ${isViolent ? 'violent' : 'safe'}">${prediction}</span>
        </div>
        <div class="result-item">
            <span class="result-label">Độ tin cậy:</span>
            <span class="result-value">${(confidence * 100).toFixed(2)}%</span>
        </div>
    `;
    webcamResult.classList.remove('hidden');
}


// ==================== HISTORY & CHARTS HANDLERS ====================

async function loadHistory() {
    const historyList = document.getElementById('history-list');
    const filter = document.getElementById('history-filter');
    if (!historyList) return;
    
    historyList.innerHTML = '<div class="loading">Đang tải dữ liệu...</div>';
    
    try {
        const source = filter ? filter.value : '';
        const url = source ? `${API_BASE}/api/history?source=${source}` : `${API_BASE}/api/history`;
        
        const response = await fetch(url);
        const data = await response.json();
        
        if (!data.success || data.detections.length === 0) {
            historyList.innerHTML = '<div class="loading">Chưa có lịch sử phát hiện nào</div>';
            return;
        }
        
        historyList.innerHTML = data.detections.map(item => {
            const isViolent = item.prediction === 'Violent';
            const date = new Date(item.timestamp).toLocaleString('vi-VN');
            return `
                <div class="history-item">
                    <div class="history-info">
                        <div class="history-name">${item.video_name || 'Camera Stream'}</div>
                        <div class="history-meta">
                            <span>🕒 ${date}</span> • 
                            <span>${item.source === 'upload' ? '📤 Upload' : '📷 Webcam'}</span>
                        </div>
                    </div>
                    <div class="history-result">
                        <div class="history-prediction ${isViolent ? 'violent' : 'safe'}">${item.prediction}</div>
                        <div class="history-confidence">${(item.confidence * 100).toFixed(1)}%</div>
                    </div>
                </div>
            `;
        }).join('');
        
        // Update charts if they exist
        updateCharts(data.detections);
        
    } catch (error) {
        console.error('History Load Error:', error);
        historyList.innerHTML = '<div class="loading" style="color: var(--danger)">Không thể tải lịch sử</div>';
    }
}

async function loadStatistics() {
    const statTotal = document.getElementById('stat-total');
    const statViolent = document.getElementById('stat-violent');
    const statSafe = document.getElementById('stat-safe');
    
    if (!statTotal) return;
    
    try {
        const response = await fetch(`${API_BASE}/api/statistics`);
        const data = await response.json();
        
        if (data.success) {
            statTotal.textContent = data.totals.total;
            statViolent.textContent = data.totals.violent;
            statSafe.textContent = data.totals.non_violent;
        }
    } catch (error) {
        console.error('Stats Error:', error);
    }
}

function updateCharts(detections) {
    if (typeof Chart === 'undefined') return;

    // 1. Pie Chart: Violent vs Non-Violent
    const pieCanvas = document.getElementById('pieChart');
    if (pieCanvas) {
        const violentCount = detections.filter(d => d.prediction === 'Violent').length;
        const safeCount = detections.length - violentCount;
        
        if (charts.pie) charts.pie.destroy();
        charts.pie = new Chart(pieCanvas, {
            type: 'doughnut',
            data: {
                labels: ['Bạo lực', 'An toàn'],
                datasets: [{
                    data: [violentCount, safeCount],
                    backgroundColor: ['#ef4444', '#10b981'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { position: 'bottom', labels: { color: '#cbd5e1' } }
                }
            }
        });
    }

    // 2. Bar Chart: Source Distribution
    const barCanvas = document.getElementById('barChart');
    if (barCanvas) {
        const uploadCount = detections.filter(d => d.source === 'upload').length;
        const webcamCount = detections.filter(d => d.source === 'webcam').length;
        
        if (charts.bar) charts.bar.destroy();
        charts.bar = new Chart(barCanvas, {
            type: 'bar',
            data: {
                labels: ['Upload Video', 'Webcam Stream'],
                datasets: [{
                    label: 'Số lượng',
                    data: [uploadCount, webcamCount],
                    backgroundColor: ['#6366f1', '#f59e0b'],
                    borderRadius: 5
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: { grid: { color: '#334155' }, ticks: { color: '#cbd5e1' } },
                    x: { grid: { display: false }, ticks: { color: '#cbd5e1' } }
                },
                plugins: { legend: { display: false } }
            }
        });
    }
    
    // 3. Line Chart: Trend over time (Simplified by Date)
    const lineCanvas = document.getElementById('timeSeriesChart');
    if (lineCanvas) {
        // Group by date
        const dateGroups = {};
        detections.forEach(d => {
            const date = new Date(d.timestamp).toLocaleDateString('vi-VN');
            dateGroups[date] = (dateGroups[date] || 0) + 1;
        });
        
        const labels = Object.keys(dateGroups).slice(-7); // Last 7 days present
        const data = labels.map(l => dateGroups[l]);
        
        if (charts.line) charts.line.destroy();
        charts.line = new Chart(lineCanvas, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Số lượng phát hiện',
                    data: data,
                    borderColor: '#6366f1',
                    backgroundColor: 'rgba(99, 102, 241, 0.2)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: { grid: { color: '#334155' }, ticks: { color: '#cbd5e1' } },
                    x: { grid: { display: false }, ticks: { color: '#cbd5e1' } }
                },
                plugins: { legend: { display: false } }
            }
        });
    }
}
