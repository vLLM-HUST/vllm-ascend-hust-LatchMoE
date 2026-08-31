TECTONIC ?= tectonic

.PHONY: all figures clean

all: main.pdf

SECTIONS := $(wildcard sections/*.tex)
FIGURES := \
	figures/latchmoe_architecture.pdf \
	figures/latchmoe_lifecycle.pdf \
	figures/motivation_characterization.pdf \
	figures/baseline_mechanisms.pdf \
	figures/capacity_sensitivity.pdf
GENERATED_TABLES := \
	generated/formal_results_macros.tex \
	generated/qualification_table.tex \
	generated/resource_table.tex \
	generated/cost_anchor_table.tex

figures: $(FIGURES)

main.pdf: main.tex $(SECTIONS) references.bib $(FIGURES) $(GENERATED_TABLES)
	$(TECTONIC) --keep-logs --keep-intermediates main.tex

clean:
	$(RM) main.aux main.bbl main.blg main.log main.out main.pdf
