# ROLOpt.jl proof-of-concept deck.
#   make           regenerate data + build slides.pdf
#   make data      run the evidence generator (data/*.dat, tables/*.tex)
#   make slides    compile slides.pdf from existing data (lualatex x2)
#   make testcount run the full test suite, capture the count for slide 5
#   make open      open slides.pdf
#   make clean     remove generated data, tables, and LaTeX aux/pdf
#
# Override the tools if they live elsewhere, e.g.:
#   make LATEX=/usr/local/texlive/2025/bin/universal-darwin/lualatex

JULIA ?= $(HOME)/.juliaup/bin/julia
LATEX ?= /Library/TeX/texbin/lualatex
PROJECT ?= ..

DATA := data/jac_taylor.dat data/precision_ladder.dat data/hk_convergence.dat data/inexact_records.dat

.PHONY: all data slides testcount open clean

all: slides

data: $(DATA)

$(DATA): generate_evidence.jl
	$(JULIA) --project=$(PROJECT) generate_evidence.jl

# pgfplots needs two passes; the deck \inputs tables/testcount.tex (kept under
# version control as a "last verified" figure, refreshed by `make testcount`).
slides: data slides.tex tables/testcount.tex
	$(LATEX) -interaction=nonstopmode -halt-on-error slides.tex
	$(LATEX) -interaction=nonstopmode -halt-on-error slides.tex

# Separate, explicit: run the suite and record the passing count + date. Not part
# of the default build (which never runs the test suite).
testcount:
	@echo "Running full ROLOpt.jl test suite ..."
	@$(JULIA) --project=$(PROJECT) -e 'using Pkg; Pkg.test()' 2>&1 | tee /tmp/rolopt_test.log; \
	N=$$(awk '/^ROLOpt\.jl/{for(i=1;i<=NF;i++) if($$i ~ /^[0-9]+$$/){print $$i; exit}}' /tmp/rolopt_test.log); \
	D=$$(date +%Y-%m-%d); \
	if [ -n "$$N" ]; then printf '\\textit{last verified: %s passing (%s)}\n' "$$N" "$$D" > tables/testcount.tex; \
	else printf '\\textit{last verified: all passing (%s)}\n' "$$D" > tables/testcount.tex; fi; \
	echo "--> $$(cat tables/testcount.tex)"

open: slides.pdf
	open slides.pdf

slides.pdf: slides

clean:
	rm -f data/*.dat tables/derivchecks.tex tables/closedform.tex \
	      tables/cvar_suppA.tex tables/inexact.tex \
	      slides.pdf slides.aux slides.log slides.nav slides.out slides.snm slides.toc slides.vrb
