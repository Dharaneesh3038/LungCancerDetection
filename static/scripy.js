document.getElementById('uploadForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    const fileInput = document.getElementById('imageUpload');
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    const resultDiv = document.getElementById('result');
    resultDiv.style.display = 'none';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        document.getElementById('predictedClass').textContent = data.class;
        document.getElementById('confidence').textContent = data.confidence;
        resultDiv.style.display = 'block';
    } catch (error) {
        alert('An error occurred. Please try again.');
        console.error(error);
    }
});