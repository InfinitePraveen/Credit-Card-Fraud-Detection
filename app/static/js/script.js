document.addEventListener('DOMContentLoaded', function () {
    // Form validation
    const form = document.getElementById('predictionForm');

    if (form) {
        form.addEventListener('submit', function (e) {
            const amount = document.getElementById('amount');
            const time = document.getElementById('time');

            // Validate amount
            if (parseFloat(amount.value) <= 0) {
                e.preventDefault();
                alert('Please enter a valid transaction amount (greater than 0).');
                amount.focus();
                return;
            }

            // Validate time
            if (parseFloat(time.value) < 0) {
                e.preventDefault();
                alert('Please enter a valid time (0 or greater).');
                time.focus();
                return;
            }
        });
    }

    // Add autocomplete suggestions for features (if needed)
    // Could be expanded to load default values from a CSV
});

// Function to analyze transaction via API (for AJAX implementation)
async function analyzeTransaction(data) {
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const result = await response.json();
        return result;
    } catch (error) {
        console.error('Error:', error);
        throw error;
    }
}

// Example usage for API endpoint
// This can be used to create a more dynamic interface
/*
document.getElementById('analyzeBtn').addEventListener('click', async function() {
    const data = {
        Time: parseFloat(document.getElementById('time').value),
        Amount: parseFloat(document.getElementById('amount').value)
    };
    
    // Add V1-V28 features
    for (let i = 1; i <= 28; i++) {
        data[`V${i}`] = parseFloat(document.getElementById(`v${i}`).value) || 0;
    }
    
    try {
        const result = await analyzeTransaction(data);
        console.log('Prediction result:', result);
        // Update UI with result
    } catch (error) {
        console.error('Prediction failed:', error);
    }
});
*/