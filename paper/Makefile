TEX = fajarquant
ARXIV_TARBALL = $(TEX)_arxiv_v3.1.tar.gz
ARXIV_WORKDIR = arxiv_v3.1

all: $(TEX).pdf

$(TEX).pdf: $(TEX).tex references.bib
	pdflatex $(TEX)
	bibtex $(TEX)
	pdflatex $(TEX)
	pdflatex $(TEX)

# arXiv submission tarball: source + .bbl + .bib, ready for upload to
# https://arxiv.org/submit. The .bbl is included so arXiv does not need
# to run bibtex itself. Verify first with `make arxiv-verify` to ensure
# the tarball compiles in isolation.
arxiv: $(ARXIV_TARBALL)

$(ARXIV_TARBALL): $(TEX).tex $(TEX).bbl references.bib
	mkdir -p $(ARXIV_WORKDIR)
	cp $(TEX).tex $(TEX).bbl references.bib $(ARXIV_WORKDIR)/
	tar czf $(ARXIV_TARBALL) -C $(ARXIV_WORKDIR) \
	    $(TEX).tex $(TEX).bbl references.bib
	@echo ""
	@echo "arXiv tarball ready: $(ARXIV_TARBALL)"
	@echo "Contents:"
	@tar tzvf $(ARXIV_TARBALL)

arxiv-verify: $(ARXIV_TARBALL)
	cd $(ARXIV_WORKDIR) && pdflatex -interaction=nonstopmode $(TEX) \
	    > /dev/null && pdflatex -interaction=nonstopmode $(TEX) \
	    > /dev/null && echo "Isolated compile OK" && \
	    pdfinfo $(TEX).pdf | grep -E "^Pages|^File size"

clean:
	rm -f $(TEX).aux $(TEX).bbl $(TEX).blg $(TEX).log $(TEX).out $(TEX).pdf $(TEX).toc
	rm -rf $(ARXIV_WORKDIR)
	rm -f $(ARXIV_TARBALL)

view: $(TEX).pdf
	xdg-open $(TEX).pdf

.PHONY: all clean view arxiv arxiv-verify
