// Frontend JavaScript logic for Callytix fraud detection system

document.addEventListener("DOMContentLoaded", () => {
    // ----------------------------------------------------
    // Drag and Drop Upload Functionality
    // ----------------------------------------------------
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const fileInfoContainer = document.getElementById("file-info-container");
    const selectedFileName = document.getElementById("selected-file-name");
    const removeFileBtn = document.getElementById("remove-file-btn");
    const predictBtn = document.getElementById("predict-btn");
    const predictionForm = document.getElementById("prediction-form");
    const predictionLoader = document.getElementById("prediction-loader");

    if (dropZone && fileInput) {
        // Highlight drop zone when item dragged over it
        ["dragenter", "dragover"].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.style.backgroundColor = "#e0f2fe";
                dropZone.style.borderColor = "var(--primary-hover)";
            }, false);
        });

        // Remove highlight when item dragged away
        ["dragleave", "drop"].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.style.backgroundColor = "var(--primary-light)";
                dropZone.style.borderColor = "var(--primary)";
            }, false);
        });

        // Handle file drop
        dropZone.addEventListener("drop", (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length > 0) {
                fileInput.files = files;
                updateFileInfo(files[0].name);
            }
        });

        // Handle file selection via click/browse
        fileInput.addEventListener("change", (e) => {
            if (fileInput.files.length > 0) {
                updateFileInfo(fileInput.files[0].name);
            }
        });

        // Update uploader UI with file details
        function updateFileInfo(filename) {
            selectedFileName.textContent = filename;
            fileInfoContainer.style.display = "flex";
            dropZone.style.display = "none";
        }

        // Handle file removal
        removeFileBtn.addEventListener("click", (e) => {
            e.preventDefault();
            fileInput.value = ""; // Clear file
            fileInfoContainer.style.display = "none";
            dropZone.style.display = "block";
        });

        // Handle prediction form submit (show loader, hide form elements)
        predictionForm.addEventListener("submit", (e) => {
            if (!fileInput.files.length) {
                alert("Please select or drop an audio file first.");
                e.preventDefault();
                return;
            }
            
            // Show loading animation
            predictionLoader.style.display = "flex";
            fileInfoContainer.style.display = "none";
            predictBtn.disabled = true;
            
            // Scroll to loader
            predictionLoader.scrollIntoView({ behavior: 'smooth' });
        });
    }

    // ----------------------------------------------------
    // Model Training Functionality (AJAX)
    // ----------------------------------------------------
    const trainBtn = document.getElementById("train-btn");
    const trainingLoader = document.getElementById("training-loader");
    const metricsContainer = document.getElementById("metrics-container");
    const modelStatusBadge = document.getElementById("model-status-badge");
    const modelStatusText = document.getElementById("model-status-text");

    if (trainBtn) {
        trainBtn.addEventListener("click", async () => {
            // Confirm model training with user
            if (confirm("Are you sure you want to train the model? This will download the Kaggle dataset if not already present, run text preprocessing, split the dataset, fit TF-IDF & Naive Bayes, and compute final metrics.")) {
                
                // Set loading state
                trainBtn.style.display = "none";
                trainingLoader.style.display = "flex";
                metricsContainer.style.display = "none";
                
                try {
                    const response = await fetch("/train", {
                        method: "POST"
                    });
                    const result = await response.json();
                    
                    if (result.success) {
                        // Update metrics details in UI
                        const metrics = result.metrics;
                        
                        document.getElementById("metric-accuracy").textContent = (metrics.accuracy * 100).toFixed(2) + "%";
                        document.getElementById("metric-precision").textContent = (metrics.precision * 100).toFixed(2) + "%";
                        document.getElementById("metric-recall").textContent = (metrics.recall * 100).toFixed(2) + "%";
                        document.getElementById("metric-f1").textContent = (metrics.f1_score * 100).toFixed(2) + "%";
                        
                        // Confusion matrix: cm[actual][predicted]
                        const cm = metrics.confusion_matrix;
                        document.getElementById("cm-0-0").textContent = cm[0][0]; // TN
                        document.getElementById("cm-0-1").textContent = cm[0][1]; // FP
                        document.getElementById("cm-1-0").textContent = cm[1][0]; // FN
                        document.getElementById("cm-1-1").textContent = cm[1][1]; // TP
                        
                        // Show metrics container
                        metricsContainer.style.display = "block";
                        
                        // Update status badge in header
                        modelStatusBadge.className = "status-badge trained";
                        modelStatusText.textContent = "Model Trained";
                        
                        // Enable predict button
                        if (predictBtn) {
                            predictBtn.disabled = false;
                        }
                        
                        alert("Model trained and validated successfully!");
                    } else {
                        alert("Training failed: " + result.message);
                    }
                } catch (err) {
                    alert("An error occurred during training: " + err.message);
                } finally {
                    // Restore train button
                    trainingLoader.style.display = "none";
                    trainBtn.style.display = "inline-flex";
                    trainBtn.textContent = "Retrain Model";
                }
            }
        });
    }
});
