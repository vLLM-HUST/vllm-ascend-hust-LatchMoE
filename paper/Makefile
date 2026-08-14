TECTONIC ?= tectonic

.PHONY: all figures clean

all: main.pdf

figures: figures/matched_ttft.pdf

figures/matched_ttft.pdf: scripts/plot_issue17_ttft.py ../docs/evidence/bundles/issue-17-matched-ttft-31621de.tar.gz
	python scripts/plot_issue17_ttft.py \
		--bundle ../docs/evidence/bundles/issue-17-matched-ttft-31621de.tar.gz \
		--output $@

SECTIONS := $(wildcard sections/*.tex)

main.pdf: main.tex $(SECTIONS) references.bib usenix-2020-09.sty figures/matched_ttft.pdf
	$(TECTONIC) --keep-logs --keep-intermediates main.tex

clean:
	$(RM) main.aux main.bbl main.blg main.log main.out main.pdf
