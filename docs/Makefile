# Build every top-level .tex into a .pdf next to it; all auxiliary files go to build/.
# Usage: make            (all documents)
#        make presentation.pdf
#        make clean      (remove build/)
TEXBIN ?= /home/software/sloan/texlive/2016/bin/x86_64-linux
export PATH := $(TEXBIN):$(PATH)

TEX  := $(wildcard *.tex)
PDF  := $(TEX:.tex=.pdf)

.PHONY: all clean
all: $(PDF)

%.pdf: %.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build $<
	cp build/$@ $@

clean:
	rm -rf build
