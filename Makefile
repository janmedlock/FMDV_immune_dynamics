# Build an HTML version of the README.
README.html: README.md
	pandoc -s -o README.html -f gfm --shift-heading-level-by=-1 README.md
