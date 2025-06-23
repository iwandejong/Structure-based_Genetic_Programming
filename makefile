compile:
	g++ -std=c++2a *.cpp -o main
alt:
	clang++ -std=c++20  *.cpp -o main
run:
	./main
leaks:
	leaks -atExit -- ./main
clean:
	rm ./main