// DOM elements
const form = document.getElementById('uploadForm');
const fileInput = document.getElementById('fileInput');
const previewImage = document.getElementById('previewImage');
const imagePreview = document.getElementById('imagePreview');
const analyzeButton = form.querySelector('button[type="submit"]');
const resultModal = document.getElementById('resultModal');
const uploadedImage = document.getElementById('uploadedImage');
const predictionText = document.getElementById('prediction');
const confidenceText = document.getElementById('confidence');
const closeResultModal = document.getElementById('closeResultModal');
const errorAlert = document.getElementById('errorAlert');
const loadingDiv = document.getElementById('loading');
const metricsButton = document.getElementById('metricsButton');
const metricsModal = document.getElementById('metricsModal');
const closeModal = document.getElementById('closeModal');
const precisionText = document.getElementById('precision');
const recallText = document.getElementById('recall');
const f1ScoreText = document.getElementById('f1Score');
const confusionMatrixTable = document.getElementById('confusionMatrix');

// Validate file
const validateFile = (file) => {
    const maxSize = 5 * 1024 * 1024; // 5MB
    const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg'];
    if (!file) return 'No file selected.';
    if (!allowedTypes.includes(file.type)) return 'Only PNG, JPG, or JPEG files are allowed.';
    if (file.size > maxSize) return 'File size exceeds 5MB.';
    return null;
};

// Show error alert
const showError = (message) => {
    errorAlert.textContent = message;
    errorAlert.classList.remove('hidden');
    setTimeout(() => errorAlert.classList.add('hidden'), 5000);
};

// Handle file input change (preview image)
fileInput.addEventListener('change', () => {
    const file = fileInput.files[0];
    const error = validateFile(file);
    if (error) {
        showError(error);
        imagePreview.classList.add('hidden');
        analyzeButton.disabled = true;
        return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        imagePreview.classList.remove('hidden');
        analyzeButton.disabled = false;
    };
    reader.readAsDataURL(file);
});

// Handle form submission
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const file = fileInput.files[0];
    const error = validateFile(file);
    if (error) {
        showError(error);
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    // Show loading and reset UI
    loadingDiv.classList.remove('hidden');
    errorAlert.classList.add('hidden');
    resultModal.classList.add('hidden');
    uploadedImage.classList.add('hidden');

    // Set uploaded image
    uploadedImage.src = previewImage.src;
    uploadedImage.classList.remove('hidden');

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData,
        });
        const data = await response.json();

        if (data.error || !response.ok) {
            showError(data.error || 'An error occurred during prediction.');
            uploadedImage.classList.add('hidden');
            return;
        }

        // Display results in modal
        predictionText.textContent = `Class: ${data.class}`;
        confidenceText.textContent = `Confidence: ${parseFloat(data.confidence).toFixed(2)}%`;
        resultModal.classList.remove('hidden');
        resultModal.classList.add('fade-in');
    } catch (error) {
        showError('Could not connect to the server.');
        uploadedImage.classList.add('hidden');
        console.error('Prediction error:', error);
    } finally {
        loadingDiv.classList.add('hidden');
    }
});

// Handle result modal close
closeResultModal.addEventListener('click', () => {
    resultModal.classList.add('hidden');
});

// Close result modal on outside click
resultModal.addEventListener('click', (e) => {
    if (e.target === resultModal) {
        resultModal.classList.add('hidden');
    }
});

// Close result modal with Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !resultModal.classList.contains('hidden')) {
        resultModal.classList.add('hidden');
    }
});

// Handle metrics button click
metricsButton.addEventListener('click', async () => {
    loadingDiv.classList.remove('hidden');
    errorAlert.classList.add('hidden');

    try {
        const response = await fetch('/metrics', {
            method: 'GET',
        });
        const data = await response.json();

        if (!response.ok || data.error) {
            showError(data.error || 'An error occurred while fetching metrics.');
            return;
        }

        // Display metrics (formatted to 4 decimal places)
        precisionText.textContent = `Precision: ${data.precision.toFixed(4)}`;
        recallText.textContent = `Recall: ${data.recall.toFixed(4)}`;
        f1ScoreText.textContent = `F1 Score: ${data.f1_score.toFixed(4)}`;

        // Display confusion matrix
        const labels = ['Benign', 'Malignant', 'Normal'];
        confusionMatrixTable.innerHTML = '';
        data.confusion_matrix.forEach((row, i) => {
            const tr = document.createElement('tr');
            tr.classList.add('border-t');
            tr.innerHTML = `
                <td class="p-2 text-left">${labels[i]}</td>
                ${row.map(val => `<td class="p-2 text-center">${val}</td>`).join('')}
            `;
            confusionMatrixTable.appendChild(tr);
        });

        metricsModal.classList.remove('hidden');
        metricsModal.classList.add('fade-in');
    } catch (error) {
        showError('Could not connect to the server.');
        console.error('Metrics error:', error);
    } finally {
        loadingDiv.classList.add('hidden');
    }
});

// Handle metrics modal close
closeModal.addEventListener('click', () => {
    metricsModal.classList.add('hidden');
});

// Close metrics modal on outside click
metricsModal.addEventListener('click', (e) => {
    if (e.target === metricsModal) {
        metricsModal.classList.add('hidden');
    }
});

// Close metrics modal with Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !metricsModal.classList.contains('hidden')) {
        metricsModal.classList.add('hidden');
    }
});