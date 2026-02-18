// Initialize page
document.addEventListener('DOMContentLoaded', function () {
    loadStatus();
    setupEventListeners();
    loadStatus();
});

// Setup event listeners
function setupEventListeners() {
    document.getElementById('uploadBtn').addEventListener('click', uploadFile);
    document.getElementById('askBtn').addEventListener('click', askQuestion);
    document.getElementById('questionInput').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') askQuestion();
    });
    document.getElementById('memoryBtn').addEventListener('click', showMemory);
    document.getElementById('clearBtn').addEventListener('click', clearIndex);

    // Modal controls
    const modal = document.getElementById('memoryModal');
    const closeBtn = document.querySelector('.close');
    closeBtn.addEventListener('click', () => modal.style.display = 'none');
    window.addEventListener('click', (e) => {
        if (e.target === modal) modal.style.display = 'none';
    });

    // Tab controls
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const tabName = e.target.dataset.tab;
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            e.target.classList.add('active');
            document.getElementById(tabName + 'Memory').classList.add('active');
        });
    });
}

// Load status
async function loadStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();

        const statusInfo = document.getElementById('statusInfo');
        statusInfo.innerHTML = `
            <div><strong>Documents:</strong> ${data.total_documents}</div>
            <div><strong>Sources:</strong> ${data.unique_sources}</div>
            ${data.sources.length > 0 ? `<div><strong>Files:</strong><br>${data.sources.join('<br>')}</div>` : ''}
        `;
    } catch (error) {
        console.error('Error loading status:', error);
    }
}

// Upload file
async function uploadFile() {
    const fileInput = document.getElementById('fileInput');
    const files = fileInput.files;

    if (files.length === 0) {
        alert('Please select a file');
        return;
    }

    const statusDiv = document.getElementById('uploadStatus');
    statusDiv.textContent = 'Uploading...';

    for (let file of files) {
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/add-document', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                statusDiv.textContent = `✓ ${data.message}`;
                fileInput.value = '';
                loadStatus();

                // Clear status after 3 seconds
                setTimeout(() => statusDiv.textContent = '', 3000);
            } else {
                statusDiv.textContent = `✗ Error: ${data.error}`;
            }
        } catch (error) {
            statusDiv.textContent = `✗ Error: ${error.message}`;
        }
    }
}

// Ask question
async function askQuestion() {
    const questionInput = document.getElementById('questionInput');
    const question = questionInput.value.trim();

    if (!question) return;

    // Add user message to chat
    addMessage('user', question);
    questionInput.value = '';

    // Add loading indicator
    const loadingId = addMessage('bot', '<div class="loading">Searching documents...</div>');

    try {
        const response = await fetch('/api/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question })
        });

        const data = await response.json();

        if (response.ok) {
            // Remove loading message
            document.getElementById(loadingId).remove();

            // Add answer
            const answerHTML = `
                <div class="answer">${escapeHtml(data.answer.trim())}</div>
                ${data.citations.length > 0 ? `
                    <div class="citations">
                        <h4>📖 Sources</h4>
                        ${data.citations.map((c, i) => `
                            <div class="citation">
                                <div class="citation-source">[${i + 1}] ${escapeHtml(c.source)} (Chunk ${c.chunk_id})</div>
                                <div class="citation-snippet">"${escapeHtml(c.snippet)}"</div>
                                <span class="citation-score">Score: ${(c.relevance_score * 100).toFixed(0)}%</span>
                            </div>
                        `).join('')}
                    </div>
                ` : ''}
            `;
            const botMessageId = addMessage('bot', answerHTML);

            // Scroll to the top of the bot message
            const botElement = document.getElementById(botMessageId);
            botElement.scrollIntoView({ behavior: "smooth", block: "start" });
        } else {
            document.getElementById(loadingId).remove();
            addMessage('bot', `<div style="color: #e74c3c;">Error: ${data.error}</div>`);
        }
    } catch (error) {
        document.getElementById(loadingId).remove();
        addMessage('bot', `<div style="color: #e74c3c;">Error: ${error.message}</div>`);
    }
}

// Add message to chat
function addMessage(sender, content) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    const messageId = 'msg-' + Date.now();
    messageDiv.id = messageId;
    messageDiv.className = `message message-${sender}`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = content;

    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);

    // Scroll to bottom
    if (sender === 'user') {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    return messageId;
}

// Show memory
async function showMemory() {
    const modal = document.getElementById('memoryModal');
    modal.style.display = 'block';

    try {
        const response = await fetch('/api/memory');
        const data = await response.json();

        document.getElementById('userMemoryContent').textContent = data.user_memory || 'No user memory yet';
        document.getElementById('companyMemoryContent').textContent = data.company_memory || 'No company memory yet';
    } catch (error) {
        document.getElementById('userMemoryContent').textContent = 'Error loading memory: ' + error.message;
        document.getElementById('companyMemoryContent').textContent = 'Error loading memory: ' + error.message;
    }
}

// Clear index
async function clearIndex() {
    if (!confirm('Are you sure you want to clear the index? This cannot be undone.')) {
        return;
    }

    try {
        const response = await fetch('/api/clear', {
            method: 'POST'
        });

        const data = await response.json();

        if (response.ok) {
            document.getElementById('chatMessages').innerHTML = '';
            loadStatus();
            alert('Index cleared successfully');
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        alert('Error: ' + error.message);
    }
}

// Utility function to escape HTML
function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}
