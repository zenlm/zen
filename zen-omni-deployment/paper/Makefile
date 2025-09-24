# Makefile for Zen-Omni Paper

LATEX = pdflatex
BIBTEX = bibtex
PAPER = zen-omni

.PHONY: all clean view

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex
	$(LATEX) $(PAPER)
	$(LATEX) $(PAPER)
	$(LATEX) $(PAPER)

clean:
	rm -f $(PAPER).aux $(PAPER).log $(PAPER).out $(PAPER).toc
	rm -f $(PAPER).bbl $(PAPER).blg
	rm -f $(PAPER).nav $(PAPER).snm
	rm -f $(PAPER).synctex.gz
	rm -f $(PAPER).pdf

view: $(PAPER).pdf
	open $(PAPER).pdf || xdg-open $(PAPER).pdf

rebuild: clean all