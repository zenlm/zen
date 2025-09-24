# Makefile for Zen-Nano paper compilation

LATEX = pdflatex
BIBTEX = bibtex
PAPER = zen-nano
CLEAN_EXTS = aux bbl blg log out toc lof lot fdb_latexmk fls synctex.gz

.PHONY: all clean paper view

all: paper

paper: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex references.bib
	@echo "🔨 Building $(PAPER).pdf..."
	@$(LATEX) $(PAPER).tex
	@$(BIBTEX) $(PAPER)
	@$(LATEX) $(PAPER).tex
	@$(LATEX) $(PAPER).tex
	@echo "✅ Paper built successfully: $(PAPER).pdf"

view: $(PAPER).pdf
	@echo "📄 Opening $(PAPER).pdf..."
	@open $(PAPER).pdf 2>/dev/null || xdg-open $(PAPER).pdf 2>/dev/null || echo "Please open $(PAPER).pdf manually"

clean:
	@echo "🧹 Cleaning temporary files..."
	@for ext in $(CLEAN_EXTS); do \
		rm -f $(PAPER).$$ext; \
	done
	@echo "✅ Cleaned"

distclean: clean
	@echo "🗑️  Removing PDF..."
	@rm -f $(PAPER).pdf
	@echo "✅ All outputs removed"

help:
	@echo "Zen-Nano Paper Build System"
	@echo "============================"
	@echo ""
	@echo "Available targets:"
	@echo "  make          - Build the paper (default)"
	@echo "  make paper    - Build the paper"
	@echo "  make view     - Build and open the PDF"
	@echo "  make clean    - Remove temporary files"
	@echo "  make distclean - Remove all generated files"
	@echo "  make help     - Show this help message"
	@echo ""
	@echo "Requirements:"
	@echo "  - pdflatex"
	@echo "  - bibtex"