# Keep auxiliary files out of docs/; the Makefile copies the finished PDF up.
$out_dir = 'build';
$pdf_mode = 1;
$pdflatex = 'pdflatex -interaction=nonstopmode -halt-on-error %O %S';
