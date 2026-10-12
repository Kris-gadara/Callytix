document.addEventListener("DOMContentLoaded", () => {
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const fileInfoContainer = document.getElementById("file-info-container");
    const selectedFileName = document.getElementById("selected-file-name");
    const removeFileBtn = document.getElementById("remove-file-btn");
    const predictionForm = document.getElementById("prediction-form");
    const predictionLoader = document.getElementById("prediction-loader");
    const predictBtn = document.getElementById("predict-btn");

    if (!dropZone || !fileInput || !fileInfoContainer || !selectedFileName) {
        return;
    }

    const updateFileInfo = (file) => {
        selectedFileName.textContent = file.name;
        fileInfoContainer.style.display = "flex";
        dropZone.style.display = "none";
    };

    ["dragenter", "dragover"].forEach((eventName) => {
        dropZone.addEventListener(eventName, (event) => {
            event.preventDefault();
            dropZone.classList.add("is-dragging");
        });
    });

    ["dragleave", "drop"].forEach((eventName) => {
        dropZone.addEventListener(eventName, (event) => {
            event.preventDefault();
            dropZone.classList.remove("is-dragging");
        });
    });

    dropZone.addEventListener("drop", (event) => {
        const files = event.dataTransfer?.files;
        if (!files || files.length === 0) {
            return;
        }

        fileInput.files = files;
        updateFileInfo(files[0]);
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) {
            updateFileInfo(fileInput.files[0]);
        }
    });

    removeFileBtn?.addEventListener("click", () => {
        fileInput.value = "";
        fileInfoContainer.style.display = "none";
        dropZone.style.display = "grid";
    });

    predictionForm?.addEventListener("submit", (event) => {
        if (fileInput.files.length === 0) {
            event.preventDefault();
            fileInput.reportValidity();
            return;
        }

        if (predictionLoader) {
            predictionLoader.style.display = "flex";
            predictionLoader.scrollIntoView({ behavior: "smooth", block: "center" });
        }
        fileInfoContainer.style.display = "none";
        if (predictBtn) {
            predictBtn.disabled = true;
            predictBtn.textContent = "Analyzing call…";
        }
    });
});
