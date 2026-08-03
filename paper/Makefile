TECTONIC ?= tectonic

.PHONY: all clean

all: main.pdf

main.pdf: main.tex references.bib usenix-2020-09.sty
	$(TECTONIC) --keep-logs --keep-intermediates main.tex

clean:
	$(RM) main.aux main.bbl main.blg main.log main.out main.pdf
