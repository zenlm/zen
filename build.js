#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

// Read templates
const header = fs.readFileSync(path.join(__dirname, 'docs/templates/header.html'), 'utf8');
const footer = fs.readFileSync(path.join(__dirname, 'docs/templates/footer.html'), 'utf8');

// Function to update active nav link
function updateActiveNav(html, page) {
    return html.replace(
        new RegExp(`<a href="${page}.html">`, 'g'),
        `<a href="${page}.html" class="active">`
    );
}

// Build pages
const pages = [
    {
        src: 'src/index.html',
        dest: 'docs/index.html',
        activePage: 'index'
    },
    {
        src: 'src/models.html',
        dest: 'docs/models.html',
        activePage: 'models'
    },
    {
        src: 'src/research.html',
        dest: 'docs/research.html',
        activePage: 'research'
    }
];

pages.forEach(({ src, dest, activePage }) => {
    if (!fs.existsSync(src)) {
        console.log(`Source file ${src} not found, skipping...`);
        return;
    }

    let content = fs.readFileSync(src, 'utf8');

    // Replace placeholders with actual header/footer and update active nav
    let headerWithActive = updateActiveNav(header, activePage);
    content = content.replace('<!-- HEADER -->', headerWithActive);
    content = content.replace('<!-- FOOTER -->', footer);

    fs.writeFileSync(dest, content, 'utf8');
    console.log(`Built ${dest}`);
});

console.log('Build complete!');
