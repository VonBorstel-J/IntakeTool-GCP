// app.js

document.addEventListener('DOMContentLoaded', () => {
    const statusIndicator = document.getElementById('status-indicator');
    const statusText = document.getElementById('status-text');
    const logsContainer = document.getElementById('logs');
    const emailIdForm = document.getElementById('email-id-form');
    const customTextForm = document.getElementById('custom-text-form');
    const progressBar = document.getElementById('progress-bar');
    const progressSteps = document.getElementById('progress-steps');
    const parsedResults = document.getElementById('parsed-results');

    // Function to update status
    async function updateStatus() {
        try {
            const response = await fetch('/api/status');
            if (!response.ok) throw new Error('Network response was not ok');
            const data = await response.json();
            statusText.textContent = data.status;
            setStatus(true);
            addLog(`Status check: ${data.status}`);
        } catch (error) {
            statusText.textContent = 'Service is down';
            setStatus(false);
            addLog('Status check failed: Using mock data.');
        }
    }

    // Function to set status indicator
    function setStatus(isActive) {
        if (isActive) {
            statusIndicator.classList.add('status-active');
            statusIndicator.classList.remove('status-error');
        } else {
            statusIndicator.classList.add('status-error');
            statusIndicator.classList.remove('status-active');
        }
    }

    // Function to add logs
    function addLog(message) {
        const logEntry = document.createElement('div');
        logEntry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
        logsContainer.appendChild(logEntry);
        logsContainer.scrollTop = logsContainer.scrollHeight;
    }

    // Function to handle form submissions
    async function handleFormSubmit(event, type) {
        event.preventDefault();
        let payload = {};
        if (type === 'email') {
            const emailId = document.getElementById('emailIdInput').value.trim();
            if (!emailId) {
                alert('Please enter a valid Email ID.');
                return;
            }
            payload = { email_id: emailId };
            addLog(`Submitting Email ID: ${emailId}`);
        } else if (type === 'custom') {
            const customText = document.getElementById('customTextInput').value.trim();
            if (!customText) {
                alert('Please enter some text to parse.');
                return;
            }
            payload = { custom_text: customText };
            addLog('Submitting Custom Text for Parsing.');
        }

        // Reset progress and parsed results
        resetProgress();
        clearParsedResults();

        // Mock backend interaction if service is down
        try {
            let response, data;
            if (type === 'email') {
                response = await fetch('/api/parse-email', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
            } else {
                response = await fetch('/api/parse-text', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
            }

            if (!response.ok) throw new Error('Network response was not ok');

            data = await response.json();
            addLog('Parsing request successful.');
            simulateProgress();
            displayParsedResults(data.parsed_data || data.results || {});
        } catch (error) {
            addLog('Backend unavailable. Using mock data.');
            simulateProgress();
            displayParsedResults(getMockParsedData());
        }
    }

    // Function to reset progress
    function resetProgress() {
        progressBar.style.width = '0%';
        progressBar.textContent = '0%';
        progressSteps.innerHTML = '';
    }

    // Function to simulate progress
    function simulateProgress() {
        const steps = ['Fetching Data', 'Parsing Data', 'Finalizing'];
        steps.forEach((step, index) => {
            const listItem = document.createElement('li');
            listItem.classList.add('list-group-item');
            listItem.textContent = step;
            progressSteps.appendChild(listItem);
        });

        let progress = 0;
        const interval = setInterval(() => {
            progress += 33;
            if (progress > 100) progress = 100;
            progressBar.style.width = `${progress}%`;
            progressBar.textContent = `${progress}%`;

            // Mark steps as completed
            for (let i = 0; i < Math.floor(progress / 33); i++) {
                progressSteps.children[i].classList.add('text-success');
            }

            if (progress >= 100) clearInterval(interval);
        }, 1000);
    }

    // Function to display parsed results
    function displayParsedResults(data) {
        parsedResults.innerHTML = '';
        if (typeof data !== 'object' || Object.keys(data).length === 0) {
            parsedResults.innerHTML = '<p>No parsed data available.</p>';
            return;
        }

        const dataStr = JSON.stringify(data, null, 2);
        const accordionItem = document.createElement('div');
        accordionItem.classList.add('accordion-item');

        const headerId = `heading-${Date.now()}`;
        const collapseId = `collapse-${Date.now()}`;

        accordionItem.innerHTML = `
            <h2 class="accordion-header" id="${headerId}">
                <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#${collapseId}" aria-expanded="false" aria-controls="${collapseId}">
                    Parsed Data
                </button>
            </h2>
            <div id="${collapseId}" class="accordion-collapse collapse" aria-labelledby="${headerId}" data-bs-parent="#parsed-results">
                <div class="accordion-body">
                    <pre>${dataStr}</pre>
                </div>
            </div>
        `;
        parsedResults.appendChild(accordionItem);
    }

    // Function to clear parsed results
    function clearParsedResults() {
        parsedResults.innerHTML = '';
    }

    // Function to get mock parsed data
    function getMockParsedData() {
        return {
            "subject": "Mock Subject",
            "sender": "mock@example.com",
            "date": "2024-04-27T12:34:56Z",
            "body": "This is a mock email body for demonstration purposes.",
            "attachments": [
                {
                    "filename": "document.pdf",
                    "url": "https://example.com/document.pdf"
                }
            ]
        };
    }

    // Event listeners for form submissions
    emailIdForm.addEventListener('submit', (e) => handleFormSubmit(e, 'email'));
    customTextForm.addEventListener('submit', (e) => handleFormSubmit(e, 'custom'));

    // Initial status check
    updateStatus();

    // Optional: Periodic status updates every 60 seconds
    setInterval(updateStatus, 60000);
});
