/**
 * PMGSY Scheme Detector - Main JavaScript
 * Handles form submissions, API calls, and UI interactions
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Form submission handler
    const predictionForm = document.getElementById('singlePredictionForm');
    if (predictionForm) {
        predictionForm.addEventListener('submit', handlePredictionSubmit);
    }

    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 80,
                    behavior: 'smooth'
                });
            }
        });
    });
});

/**
 * Handle prediction form submission
 * @param {Event} e - Form submit event
 */
async function handlePredictionSubmit(e) {
    e.preventDefault();
    
    const form = e.target;
    const submitBtn = form.querySelector('button[type="submit"]');
    const originalBtnText = submitBtn.innerHTML;
    
    // Show loading state
    submitBtn.disabled = true;
    submitBtn.innerHTML = `
        <span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>
        Processing...
    `;
    
    try {
        // Get form data
        const formData = {
            projectName: document.getElementById('projectName').value,
            financialData: parseFloat(document.getElementById('financialData').value),
            physicalProgress: parseFloat(document.getElementById('physicalProgress').value)
        };
        
        // Call prediction API
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });
        
        if (!response.ok) {
            throw new Error('Prediction failed');
        }
        
        const result = await response.json();
        
        // Display results
        displayPredictionResult(result);
        
    } catch (error) {
        console.error('Prediction error:', error);
        showAlert('An error occurred while processing your request. Please try again.', 'danger');
    } finally {
        // Reset button state
        submitBtn.disabled = false;
        submitBtn.innerHTML = originalBtnText;
    }
}

/**
 * Display prediction results
 * @param {Object} result - Prediction result object
 */
function displayPredictionResult(result) {
    const resultDiv = document.getElementById('predictionResult');
    const schemeElement = document.getElementById('predictedScheme');
    const confidenceElement = document.getElementById('predictionConfidence');
    
    if (result.error) {
        showAlert(result.error, 'danger');
        return;
    }
    
    // Update UI with prediction
    schemeElement.textContent = result.prediction || 'N/A';
    confidenceElement.textContent = result.confidence ? result.confidence.toFixed(1) : 'N/A';
    
    // Show result section with animation
    resultDiv.classList.remove('d-none');
    resultDiv.classList.add('show');
    
    // Scroll to results
    resultDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

/**
 * Show alert message
 * @param {string} message - Message to display
 * @param {string} type - Alert type (success, danger, warning, info)
 */
function showAlert(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.role = 'alert';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    // Add to the top of the page
    const container = document.querySelector('.container:first-of-type');
    container.insertBefore(alertDiv, container.firstChild);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        const alert = bootstrap.Alert.getOrCreateInstance(alertDiv);
        if (alert) {
            alert.close();
        }
    }, 5000);
}

/**
 * Format number with commas
 * @param {number} num - Number to format
 * @returns {string} Formatted number string
 */
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

// Export functions for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        handlePredictionSubmit,
        displayPredictionResult,
        showAlert,
        formatNumber
    };
}
