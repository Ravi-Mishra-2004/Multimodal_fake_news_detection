// Handle image preview
document.getElementById('image').addEventListener('change', function(e) {
    const file = e.target.files[0];
    const preview = document.getElementById('imagePreview');
    
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            preview.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
        };
        reader.readAsDataURL(file);
    } else {
        preview.innerHTML = '';
    }
});

// Handle form submission
document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const formData = new FormData(this);
    const submitBtn = document.getElementById('submitBtn');
    const resultDiv = document.getElementById('result');
    const errorDiv = document.getElementById('error');
    const resultContent = document.getElementById('resultContent');
    const errorMessage = document.getElementById('errorMessage');
    
    // Hide previous results
    resultDiv.style.display = 'none';
    errorDiv.style.display = 'none';
    
    // Show loading state
    submitBtn.disabled = true;
    submitBtn.querySelector('.btn-text').style.display = 'none';
    submitBtn.querySelector('.btn-loader').style.display = 'inline';
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok && data.success) {
            // Display results
            const isFake = data.prediction === 'Fake';
            const predictionClass = isFake ? 'fake' : 'real';
            
            resultContent.innerHTML = `
                <div class="prediction-box ${predictionClass}">
                    <div class="prediction-label">${data.prediction}</div>
                    <div class="confidence">Confidence: ${data.confidence}%</div>
                </div>
                <div class="probabilities">
                    <div class="prob-box real">
                        <div class="prob-label">Real News</div>
                        <div class="prob-value">${data.real_probability}%</div>
                    </div>
                    <div class="prob-box fake">
                        <div class="prob-label">Fake News</div>
                        <div class="prob-value">${data.fake_probability}%</div>
                    </div>
                </div>
            `;
            
            resultDiv.style.display = 'block';
        } else {
            // Display error
            errorMessage.textContent = data.error || 'An error occurred';
            errorDiv.style.display = 'block';
        }
    } catch (error) {
        errorMessage.textContent = 'Network error: ' + error.message;
        errorDiv.style.display = 'block';
    } finally {
        // Reset button state
        submitBtn.disabled = false;
        submitBtn.querySelector('.btn-text').style.display = 'inline';
        submitBtn.querySelector('.btn-loader').style.display = 'none';
    }
});

