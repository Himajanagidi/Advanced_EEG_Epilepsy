// Main JavaScript functionality
document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // File upload handling
    initializeFileUpload();
    
    // Smooth scrolling
    initializeSmoothScrolling();
    
    // Animation on scroll
    initializeScrollAnimations();
});

function initializeFileUpload() {
    const fileInput = document.getElementById('file');
    const uploadArea = document.querySelector('.file-upload-area');
    const fileInfo = document.getElementById('file-info');
    
    if (!fileInput || !uploadArea) return;

    // Drag and drop functionality
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            displayFileInfo(files[0]);
        }
    });

    // Click to upload
    uploadArea.addEventListener('click', function() {
        fileInput.click();
    });

    // File input change
    fileInput.addEventListener('change', function() {
        if (fileInput.files.length > 0) {
            displayFileInfo(fileInput.files[0]);
        }
    });

    function displayFileInfo(file) {
        if (fileInfo) {
            fileInfo.innerHTML = `
                <div class="alert alert-info">
                    <i class="fas fa-file-csv me-2"></i>
                    <strong>Selected:</strong> ${file.name} (${formatFileSize(file.size)})
                </div>
            `;
        }
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
}

function initializeSmoothScrolling() {
    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

function initializeScrollAnimations() {
    // Intersection Observer for animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
            }
        });
    }, observerOptions);

    // Observe elements with animation classes
    document.querySelectorAll('.custom-card, .stat-card, .chart-container').forEach(el => {
        observer.observe(el);
    });
}

// Chart creation functions
function createPieChart(containerId, data) {
    const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: 'white', size: 12 },
        showlegend: true,
        legend: {
            bgcolor: 'rgba(0,0,0,0.5)',
            font: { color: 'white' }
        }
    };

    Plotly.newPlot(containerId, data, layout, {
        responsive: true,
        displayModeBar: false
    });
}

function createBarChart(containerId, data) {
    const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(26,28,36,0.8)',
        font: { color: 'white' },
        xaxis: { gridcolor: '#444' },
        yaxis: { gridcolor: '#444' },
        showlegend: true,
        legend: {
            bgcolor: 'rgba(0,0,0,0.5)',
            font: { color: 'white' }
        }
    };

    Plotly.newPlot(containerId, data, layout, {
        responsive: true,
        displayModeBar: false
    });
}

function createLineChart(containerId, data) {
    const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(26,28,36,0.8)',
        font: { color: 'white' },
        xaxis: { 
            gridcolor: '#444',
            title: 'Time Points'
        },
        yaxis: { 
            gridcolor: '#444',
            title: 'Amplitude'
        },
        showlegend: true,
        legend: {
            bgcolor: 'rgba(0,0,0,0.5)',
            font: { color: 'white' }
        }
    };

    Plotly.newPlot(containerId, data, layout, {
        responsive: true,
        displayModeBar: false
    });
}

// Utility functions
function showLoading(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = '<div class="text-center"><div class="loading"></div><p class="mt-3">Processing...</p></div>';
    }
}

function hideLoading(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = '';
    }
}

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show`;
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    const container = document.querySelector('.flash-container') || document.body;
    container.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, 5000);
}

// Export functions for use in other scripts
window.EEGApp = {
    createPieChart,
    createBarChart,
    createLineChart,
    showLoading,
    hideLoading,
    showNotification
};
