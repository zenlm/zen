// Simple include system for header/footer
document.addEventListener('DOMContentLoaded', async function() {
    // Load header
    const headerPlaceholder = document.getElementById('header-placeholder');
    if (headerPlaceholder) {
        try {
            const response = await fetch('/templates/header.html');
            const html = await response.text();
            headerPlaceholder.outerHTML = html;

            // Set active nav link
            const currentPage = window.location.pathname.split('/').pop() || 'index.html';
            document.querySelectorAll('.nav-links a').forEach(link => {
                if (link.getAttribute('href') === currentPage) {
                    link.classList.add('active');
                }
            });
        } catch (error) {
            console.error('Error loading header:', error);
        }
    }

    // Load footer
    const footerPlaceholder = document.getElementById('footer-placeholder');
    if (footerPlaceholder) {
        try {
            const response = await fetch('/templates/footer.html');
            const html = await response.text();
            footerPlaceholder.outerHTML = html;
        } catch (error) {
            console.error('Error loading footer:', error);
        }
    }
});
