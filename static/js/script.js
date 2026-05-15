// ── Drop-zone & upload form behaviour ──────────────────────────────────────

(function () {
    const dropZone    = document.getElementById('dropZone');
    const imageInput  = document.getElementById('imageInput');
    const dzIdle      = document.getElementById('dzIdle');
    const dzPreview   = document.getElementById('dzPreview');
    const previewImg  = document.getElementById('previewImg');
    const previewName = document.getElementById('previewName');
    const analyzeBtn  = document.getElementById('analyzeBtn');
    const btnLabel    = document.getElementById('btnLabel');
    const btnSpinner  = document.getElementById('btnSpinner');
    const uploadForm  = document.getElementById('uploadForm');

    if (!dropZone) return; // Only runs on upload page

    // Drag events
    dropZone.addEventListener('dragover', function (e) {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    ['dragleave', 'dragend'].forEach(function (evt) {
        dropZone.addEventListener(evt, function () {
            dropZone.classList.remove('drag-over');
        });
    });

    dropZone.addEventListener('drop', function (e) {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            imageInput.files = files;
            showPreview(files[0]);
        }
    });

    // File input change (native picker)
    imageInput.addEventListener('change', function () {
        if (imageInput.files.length > 0) {
            showPreview(imageInput.files[0]);
        }
    });

    function showPreview(file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            previewImg.src = e.target.result;
            previewName.textContent = file.name + ' (' + (file.size / 1024).toFixed(1) + ' KB)';
            dzIdle.style.display    = 'none';
            dzPreview.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }

    // Show spinner on submit
    if (uploadForm) {
        uploadForm.addEventListener('submit', function () {
            if (imageInput.files.length > 0) {
                analyzeBtn.disabled    = true;
                btnLabel.style.display  = 'none';
                btnSpinner.style.display = 'inline-flex';
            }
        });
    }
})();
