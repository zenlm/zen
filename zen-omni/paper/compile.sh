#!/bin/bash

# Zen1-Omni Paper Compilation Script
# Compiles the LaTeX paper with proper bibliography and references

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if pdflatex is installed
if ! command -v pdflatex &> /dev/null; then
    print_error "pdflatex not found. Please install TeX distribution (e.g., TeXLive, MacTeX)"
    exit 1
fi

# Check if bibtex is installed
if ! command -v bibtex &> /dev/null; then
    print_warning "bibtex not found. Bibliography may not compile correctly"
fi

# Clean previous builds
print_status "Cleaning previous builds..."
rm -f *.aux *.log *.out *.toc *.bbl *.blg *.pdf 2>/dev/null

# First compilation
print_status "First LaTeX compilation..."
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
if [ $? -ne 0 ]; then
    print_error "First compilation failed. Check main.log for details"
    tail -20 main.log
    exit 1
fi

# Run BibTeX if bibliography exists
if [ -f references.bib ]; then
    print_status "Processing bibliography..."
    bibtex main > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        print_warning "BibTeX processing failed. Bibliography may be incomplete"
    fi
fi

# Second compilation for references
print_status "Second LaTeX compilation (resolving references)..."
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
if [ $? -ne 0 ]; then
    print_error "Second compilation failed. Check main.log for details"
    tail -20 main.log
    exit 1
fi

# Third compilation to ensure all references are resolved
print_status "Final LaTeX compilation..."
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
if [ $? -ne 0 ]; then
    print_error "Final compilation failed. Check main.log for details"
    tail -20 main.log
    exit 1
fi

# Check if PDF was created
if [ -f main.pdf ]; then
    print_status "✅ Successfully compiled: main.pdf"
    
    # Get PDF info
    pages=$(pdfinfo main.pdf 2>/dev/null | grep Pages | awk '{print $2}')
    size=$(ls -lh main.pdf | awk '{print $5}')
    
    print_status "PDF Details:"
    echo "  - Pages: $pages"
    echo "  - Size: $size"
    
    # Optional: Open PDF (macOS)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        print_status "Opening PDF..."
        open main.pdf
    fi
else
    print_error "PDF creation failed"
    exit 1
fi

# Clean auxiliary files (optional)
print_status "Cleaning auxiliary files..."
rm -f *.aux *.log *.out *.toc *.bbl *.blg 2>/dev/null

print_status "✨ Compilation complete!"