figure.png:	figure.pdf
	pdftocairo -png -r 300 -singlefile $<

figure.pdf:	figure.tex plot_figure.pdf notes/diagram.tex
	pdflatex --synctex=15 $<
