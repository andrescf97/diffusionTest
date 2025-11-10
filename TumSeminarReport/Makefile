
%.pdf:  %.tex techreportacmart.cls ACM-Reference-Format.bst
	pdflatex $<
	- bibtex $*
	pdflatex $<
	pdflatex $<
	while ( grep -q '^LaTeX Warning: Label(s) may have changed' $*.log) \
	do pdflatex $<; done

sample-acmtog.pdf:

clean:
	rm -f comment.cut sample-acmtog.{aux,bbl,blg,log,out,pdf}
	rm -f *-converted-to.pdf

