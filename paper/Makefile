all: supplement.pdf

supplement.pdf: supplement.tex FORCE
	pdflatex supplement
	bibtex supplement
	pdflatex supplement

FORCE: 
