slow:
	clang++ -Wall -Wextra -Werror -O2 -o slow slow.cpp

fast:
	nvcc fast.cu -O2 -o fast

# make in WSL sometimes doesn't detect that fast.cu changed...
clean:
	rm -f slow fast