// Upload functionality for deepfake detector

document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const uploadForm = document.getElementById('uploadForm');
    const fileInfo = document.getElementById('fileInfo');
    const fileName = document.getElementById('fileName');
    const fileDetails = document.getElementById('fileDetails');
    const submitBtn = document.getElementById('submitBtn');
    const uploadProgress = document.getElementById('uploadProgress');
    const dropZoneContent = document.getElementById('dropZoneContent');
    
    // Initialize animations
    initializeAnimations();
    
    // Counter animation for stats
    animateCounters();

    // File type mappings
    const fileTypes = {
        // Video files
        'mp4': { type: 'video', icon: 'video', color: 'primary' },
        'avi': { type: 'video', icon: 'video', color: 'primary' },
        'mov': { type: 'video', icon: 'video', color: 'primary' },
        'mkv': { type: 'video', icon: 'video', color: 'primary' },
        // Audio files
        'wav': { type: 'audio', icon: 'headphones', color: 'info' },
        'mp3': { type: 'audio', icon: 'headphones', color: 'info' },
        'm4a': { type: 'audio', icon: 'headphones', color: 'info' },
        'flac': { type: 'audio', icon: 'headphones', color: 'info' },
        // Image files
        'jpg': { type: 'image', icon: 'image', color: 'success' },
        'jpeg': { type: 'image', icon: 'image', color: 'success' },
        'png': { type: 'image', icon: 'image', color: 'success' },
        'bmp': { type: 'image', icon: 'image', color: 'success' },
        'tiff': { type: 'image', icon: 'image', color: 'success' },
        'webp': { type: 'image', icon: 'image', color: 'success' }
    };

    // Drag and drop handlers
    dropZone.addEventListener('dragover', handleDragOver);
    dropZone.addEventListener('dragenter', handleDragEnter);
    dropZone.addEventListener('dragleave', handleDragLeave);
    dropZone.addEventListener('drop', handleDrop);
    dropZone.addEventListener('click', () => fileInput.click());

    // File input change handler
    fileInput.addEventListener('change', handleFileSelect);

    // Form submit handler
    uploadForm.addEventListener('submit', handleFormSubmit);

    function handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function handleDragEnter(e) {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add('dragover');
    }

    function handleDragLeave(e) {
        e.preventDefault();
        e.stopPropagation();
        
        // Only remove dragover if we're leaving the dropZone entirely
        if (!dropZone.contains(e.relatedTarget)) {
            dropZone.classList.remove('dragover');
        }
    }

    function handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            handleFileSelect();
        }
    }

    function handleFileSelect() {
        const file = fileInput.files[0];
        if (!file) {
            hideFileInfo();
            return;
        }

        // Validate file
        const validation = validateFile(file);
        if (!validation.valid) {
            showError(validation.message);
            hideFileInfo();
            return;
        }

        // Show file information
        showFileInfo(file);
    }

    function validateFile(file) {
        // Check file size (100MB limit)
        const maxSize = 100 * 1024 * 1024; // 100MB
        if (file.size > maxSize) {
            return {
                valid: false,
                message: 'File size must be less than 100MB'
            };
        }

        // Check file type
        const extension = file.name.split('.').pop().toLowerCase();
        if (!fileTypes[extension]) {
            return {
                valid: false,
                message: 'Unsupported file type. Please upload video (MP4, AVI, MOV, MKV), audio (WAV, MP3, M4A, FLAC), or image (JPG, PNG, BMP, TIFF, WEBP) files.'
            };
        }

        return { valid: true };
    }

    function showFileInfo(file) {
        const extension = file.name.split('.').pop().toLowerCase();
        const fileType = fileTypes[extension];
        
        // Update file info display
        fileName.textContent = file.name;
        fileDetails.innerHTML = `
            <span class="badge bg-${fileType.color} me-2">
                <i data-feather="${fileType.icon}" class="me-1" style="width: 12px; height: 12px;"></i>
                ${fileType.type.toUpperCase()}
            </span>
            ${formatFileSize(file.size)} • ${getFileExtension(file.name)}
        `;

        // Show file info card
        fileInfo.style.display = 'block';
        
        // Enable submit button
        submitBtn.disabled = false;
        
        // Re-initialize feather icons
        feather.replace();
    }

    function hideFileInfo() {
        fileInfo.style.display = 'none';
        submitBtn.disabled = true;
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }

    function getFileExtension(filename) {
        return filename.split('.').pop().toUpperCase();
    }

    function showError(message) {
        // Create and show error alert
        const alertDiv = document.createElement('div');
        alertDiv.className = 'alert alert-danger alert-dismissible fade show mt-3';
        alertDiv.innerHTML = `
            <i data-feather="alert-circle" class="me-2"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        dropZone.parentNode.insertBefore(alertDiv, dropZone.nextSibling);
        
        // Re-initialize feather icons
        feather.replace();
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }

    function handleFormSubmit(e) {
        const file = fileInput.files[0];
        if (!file) {
            e.preventDefault();
            showError('Please select a file to upload');
            return;
        }

        // Show upload progress
        showUploadProgress();
        
        // Disable form elements
        submitBtn.disabled = true;
        fileInput.disabled = true;
        
        // Update submit button text
        submitBtn.innerHTML = `
            <div class="analysis-spinner"></div>
            Analyzing File...
        `;
    }

    function showUploadProgress() {
        dropZoneContent.style.display = 'none';
        uploadProgress.style.display = 'block';
        
        // Simulate progress
        const progressBar = uploadProgress.querySelector('.progress-bar');
        let progress = 0;
        
        const progressInterval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress > 95) {
                progress = 95; // Keep at 95% until actual upload completes
                clearInterval(progressInterval);
            }
            progressBar.style.width = progress + '%';
        }, 200);
    }

    // Utility function to reset form
    function resetForm() {
        fileInput.value = '';
        hideFileInfo();
        dropZoneContent.style.display = 'block';
        uploadProgress.style.display = 'none';
        submitBtn.disabled = true;
        submitBtn.innerHTML = `
            <i data-feather="zap" class="me-2"></i>
            Analyze for Deepfakes
        `;
        feather.replace();
    }

    // Add keyboard support for accessibility
    dropZone.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            fileInput.click();
        }
    });

    // Make drop zone focusable
    dropZone.setAttribute('tabindex', '0');
    dropZone.setAttribute('role', 'button');
    dropZone.setAttribute('aria-label', 'Click or drag files to upload');

    // Handle file input focus
    fileInput.addEventListener('focus', () => {
        dropZone.style.outline = '2px solid var(--bs-primary)';
    });

    fileInput.addEventListener('blur', () => {
        dropZone.style.outline = 'none';
    });
    
    // Initialize feature cards animation
    function initializeAnimations() {
        // Add entrance animations to cards
        const cards = document.querySelectorAll('.feature-card');
        const observer = new IntersectionObserver((entries) => {
            entries.forEach((entry, index) => {
                if (entry.isIntersecting) {
                    setTimeout(() => {
                        entry.target.style.opacity = '1';
                        entry.target.style.transform = 'translateY(0)';
                    }, index * 100);
                }
            });
        }, { threshold: 0.1 });
        
        cards.forEach(card => {
            card.style.opacity = '0';
            card.style.transform = 'translateY(30px)';
            card.style.transition = 'all 0.6s ease';
            observer.observe(card);
        });
        
        // Add floating animation to icons
        const icons = document.querySelectorAll('[data-feather="shield"]');
        icons.forEach(icon => {
            icon.classList.add('float-animation');
        });
    }
    
    // Animate counter numbers
    function animateCounters() {
        const counters = document.querySelectorAll('.stats-counter');
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const counter = entry.target;
                    const target = counter.textContent;
                    
                    // Extract number from text
                    const match = target.match(/([0-9.]+)/);
                    if (match) {
                        const targetNum = parseFloat(match[1]);
                        const suffix = target.replace(match[1], '');
                        
                        animateNumber(counter, 0, targetNum, suffix, 2000);
                    }
                }
            });
        }, { threshold: 0.5 });
        
        counters.forEach(counter => observer.observe(counter));
    }
    
    // Number animation function
    function animateNumber(element, start, end, suffix, duration) {
        const startTime = performance.now();
        
        function update(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            // Easing function
            const easeOut = 1 - Math.pow(1 - progress, 3);
            const current = start + (end - start) * easeOut;
            
            if (suffix.includes('%')) {
                element.textContent = current.toFixed(1) + suffix;
            } else if (suffix.includes('s')) {
                element.textContent = '<' + Math.ceil(current) + suffix;
            } else {
                element.textContent = Math.ceil(current) + suffix;
            }
            
            if (progress < 1) {
                requestAnimationFrame(update);
            }
        }
        
        requestAnimationFrame(update);
    }
    
    // Enhanced file selection with animation
    function showFileInfoEnhanced(file) {
        showFileInfo(file);
        
        // Add pulse effect to submit button
        setTimeout(() => {
            submitBtn.classList.add('pulse-glow');
        }, 500);
    }
    
    // Override the original showFileInfo to use enhanced version
    const originalShowFileInfo = showFileInfo;
    showFileInfo = showFileInfoEnhanced;
    
    // Add ripple effect to buttons
    document.querySelectorAll('.btn').forEach(button => {
        button.addEventListener('click', function(e) {
            const ripple = document.createElement('span');
            const rect = this.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            const x = e.clientX - rect.left - size / 2;
            const y = e.clientY - rect.top - size / 2;
            
            ripple.style.width = ripple.style.height = size + 'px';
            ripple.style.left = x + 'px';
            ripple.style.top = y + 'px';
            ripple.classList.add('ripple');
            
            this.appendChild(ripple);
            
            setTimeout(() => {
                ripple.remove();
            }, 600);
        });
    });
    
    // Add CSS for ripple effect
    const style = document.createElement('style');
    style.textContent = `
        .btn {
            position: relative;
            overflow: hidden;
        }
        
        .ripple {
            position: absolute;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.3);
            transform: scale(0);
            animation: ripple-animation 0.6s linear;
            pointer-events: none;
        }
        
        @keyframes ripple-animation {
            to {
                transform: scale(4);
                opacity: 0;
            }
        }
    `;
    document.head.appendChild(style);
});
