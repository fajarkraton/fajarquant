TEX = fajarquant
all: $(TEX).pdf

$(TEX).pdf: $(TEX).tex references.bib
	pdflatex $(TEX)
	bibtex $(TEX)
	pdflatex $(TEX)
	pdflatex $(TEX)

clean:
	rm -f $(TEX).aux $(TEX).bbl $(TEX).blg $(TEX).log $(TEX).out $(TEX).pdf $(TEX).toc

view: $(TEX).pdf
	xdg-open $(TEX).pdf

.PHONY: all clean view
