INC = nnSparrow/*.hpp

example: example.cpp mnist_parser.h Makefile $(INC)
	g++ -O4 example.cpp -o example
