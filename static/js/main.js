// --- SCROLL LOGIC ---
function scrollToDiagnosis() {
    document.getElementById('diagnosis').scrollIntoView({ behavior: 'smooth' });
}

// --- WIZARD LOGIC ---
let currentStep = 1;

function nextStep(step) {
    if (!validateStep(currentStep)) return;

    // Hide Current
    document.getElementById(`step-${currentStep}`).classList.remove('active');
    document.getElementById(`p-step-${currentStep}`).classList.remove('active');

    // Show Next
    currentStep = step;
    document.getElementById(`step-${currentStep}`).classList.add('active');
    document.getElementById(`p-step-${currentStep}`).classList.add('active');
}

function prevStep(step) {
    // Hide Current
    document.getElementById(`step-${currentStep}`).classList.remove('active');
    document.getElementById(`p-step-${currentStep}`).classList.remove('active');

    // Show Previous
    currentStep = step;
    document.getElementById(`step-${currentStep}`).classList.add('active');
    document.getElementById(`p-step-${currentStep}`).classList.add('active');
}

function validateStep(step) {
    const stepDiv = document.getElementById(`step-${step}`);
    const inputs = stepDiv.querySelectorAll('input, select');
    let valid = true;

    inputs.forEach(input => {
        if (!input.value) {
            input.style.borderColor = '#ef4444';
            valid = false;
        } else {
            input.style.borderColor = 'rgba(255, 255, 255, 0.1)';
        }
    });
    return valid;
}

// --- API & RESULT LOGIC ---
async function submitData() {
    if (!validateStep(3)) return;

    // Show Result Panel (Loading State)
    const resultPanel = document.getElementById('result-panel');
    resultPanel.classList.remove('hidden');
    resultPanel.classList.remove('danger-state', 'safe-state');
    
    // Gather Data
    const formData = {
        age: document.getElementById('age').value,
        sex: document.getElementById('sex').value,
        cp: document.getElementById('cp').value,
        trestbps: document.getElementById('trestbps').value,
        chol: document.getElementById('chol').value,
        fbs: document.getElementById('fbs').value,
        restecg: document.getElementById('restecg').value,
        thalach: document.getElementById('thalach').value,
        exang: document.getElementById('exang').value,
        oldpeak: document.getElementById('oldpeak').value,
        slope: document.getElementById('slope').value,
        ca: document.getElementById('ca').value,
        thal: document.getElementById('thal').value
    };

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });
        const data = await response.json();

        // Simulate calculation time for "WOW" effect
        setTimeout(() => {
            renderResult(data);
        }, 2000);

    } catch (err) {
        console.error(err);
        alert("Server Error");
        resultPanel.classList.add('hidden');
    }
}

function renderResult(data) {
    const iconBox = document.getElementById('res-icon');
    const title = document.getElementById('res-title');
    const desc = document.getElementById('res-desc');
    const percent = document.getElementById('res-percent');
    const meter = document.getElementById('meter-fill');
    const panel = document.getElementById('result-panel');

    // Stop scan animation
    document.querySelector('.scan-line').style.display = 'none';

    // Animate Bar
    meter.style.width = `${data.probability}%`;
    percent.innerText = `${data.probability}% Risk`;

    if (data.prediction === 1) {
        // High Risk
        panel.classList.add('danger-state');
        iconBox.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i>';
        title.innerText = "High Risk Detected";
        desc.innerText = "The analysis indicates a strong likelihood of heart disease. Immediate medical consultation is advised.";
    } else {
        // Low Risk
        panel.classList.add('safe-state');
        iconBox.innerHTML = '<i class="fa-solid fa-shield-heart"></i>';
        title.innerText = "Healthy Profile";
        desc.innerText = "Your vitals are within the healthy range. Continue maintaining a healthy lifestyle.";
    }
}

function resetTool() {
    location.reload();
}