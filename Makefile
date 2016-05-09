INC = nnSparrow/*.hpp

example: example.cpp mnist_parser.h Makefile $(INC)
	g++ -O4 example.cpp -o example

example_load_model: example_load_model.cpp mnist_parser.h Makefile $(INC)
		g++ -O4 example_load_model.cpp -o example_load_model
