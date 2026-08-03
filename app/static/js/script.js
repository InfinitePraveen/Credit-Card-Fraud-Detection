document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('predictionForm');

    if (form) {
        // Form validation
        form.addEventListener('submit', function (e) {
            const amount = document.getElementById('amount');
            const time = document.getElementById('time');
            let isValid = true;

            // Validate amount
            if (!amount.value || parseFloat(amount.value) <= 0) {
                e.preventDefault();
                showError(amount, 'Please enter a valid amount greater than 0');
                isValid = false;
            } else {
                clearError(amount);
            }

            // Validate time
            if (!time.value || parseFloat(time.value) < 0) {
                e.preventDefault();
                showError(time, 'Please enter a valid time (0 or greater)');
                isValid = false;
            } else {
                clearError(time);
            }

            if (isValid) {
                // Show loading state
                const submitBtn = form.querySelector('.btn-primary');
                submitBtn.innerHTML = '⏳ Analyzing...';
                submitBtn.disabled = true;
            }
        });

        // Real-time validation
        const inputs = form.querySelectorAll('input');
        inputs.forEach(input => {
            input.addEventListener('input', function () {
                if (this.value) {
                    clearError(this);
                }
            });
        });
    }
});

function showError(input, message) {
    input.style.borderColor = '#fc8181';
    input.style.backgroundColor = '#fff5f5';

    // Remove existing error message
    const existingError = input.parentElement.querySelector('.error-message');
    if (existingError) {
        existingError.remove();
    }

    // Add error message
    const error = document.createElement('div');
    error.className = 'error-message';
    error.style.cssText = `
        color: #e53e3e;
        font-size: 0.85rem;
        margin-top: 4px;
        font-weight: 500;
    `;
    error.textContent = '❌ ' + message;
    input.parentElement.appendChild(error);
}

function clearError(input) {
    input.style.borderColor = '';
    input.style.backgroundColor = '';
    const error = input.parentElement.querySelector('.error-message');
    if (error) {
        error.remove();
    }
}

// API function for AJAX calls
async function analyzeTransaction(time, amount) {
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ time, amount })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Network response was not ok');
        }

        return await response.json();
    } catch (error) {
        console.error('Error:', error);
        throw error;
    }
}

// Example of how to use the API programmatically
// This can be used if you want to build a dynamic single-page app
/*
document.getElementById('analyzeBtn').addEventListener('click', async function() {
    const time = parseFloat(document.getElementById('time').value);
    const amount = parseFloat(document.getElementById('amount').value);
    
    try {
        const result = await analyzeTransaction(time, amount);
        console.log('Prediction result:', result);
        // Update UI with result dynamically
        displayResult(result);
    } catch (error) {
        alert('Error making prediction: ' + error.message);
    }
});

function displayResult(result) {
    // Custom function to display results without page reload
    // This would be used for a single-page application
}
*/