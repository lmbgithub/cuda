# CUDA Vector Addition

This is a simple CUDA program that demonstrates vector addition using both CPU and GPU implementations.

## Requirements

- CUDA Toolkit 11.0 or higher
- CMake 3.8 or higher
- C++ compiler with C++11 support

## Building the Project

1. Create a build directory:
```bash
mkdir build
cd build
```

2. Configure and build the project:
```bash
cmake ..
make
```

## Running the Program

After building, you can run the program from the build directory:
```bash
./vector_add
```

The program will:
1. Generate two random vectors of size 1,000,000
2. Perform vector addition on both CPU and GPU
3. Compare the results
4. Display timing information and speedup

## Expected Output

The program will output:
- Vector size
- GPU execution time
- CPU execution time
- Speedup achieved
- Whether the results are correct

## Notes

- The program uses 256 threads per block
- The vector size is set to 1,000,000 elements
- Results are verified by comparing CPU and GPU outputs
