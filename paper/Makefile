all: supplement.pdf

push:
	cp supplement.pdf ../data/

supplement.pdf: supplement.tex FORCE
	pdflatex supplement
	bibtex supplement
	pdflatex supplement

FORCE: 
