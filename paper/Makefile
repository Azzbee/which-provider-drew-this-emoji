# Build the paper:  make            (needs a TeX Live with latexmk, or use Overleaf: upload this folder, pdfLaTeX)
# Rebuild figures:  make figures    (needs the codebase outputs; figures 1, 3, 12 also need the Kaggle images)
all:
	latexmk -pdf -interaction=nonstopmode main.tex
	cp main.pdf Paper_The_Winners.pdf
figures:
	python make_figures.py
clean:
	latexmk -C
