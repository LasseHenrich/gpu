slow: slow_scan.cu
	nvcc -Werror all-warnings -lcurand -O2 -o slow_scan slow_scan.cu

fast: fast_scan.cu
	nvcc -Werror all-warnings -lcurand -O2 -o fast_scan fast_scan.cu

# make in WSL sometimes doesn't detect that fast.cu changed...
clean:
	rm -f slow fast

# clean and rebuild
cf: clean fast
	