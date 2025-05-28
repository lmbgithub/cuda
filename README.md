# CUDA Vector Addition

This repository contains CUDA implementations of vector addition, demonstrating parallel computing concepts and GPU acceleration.

## Project Structure

```
src/
├── 1_vector_addition/
│   ├── vector_add.cu      # Main implementation
│   ├── CMakeLists.txt     # Build configuration
│   └── README.md          # Implementation details
```

## Requirements

- CUDA Toolkit (version 11.0 or higher)
- CMake (version 3.20 or higher)
- C++ compiler with C++20 support
- NVIDIA GPU with compute capability 5.0 or higher

## Building the Project

1. Create a build directory:
```bash
mkdir build
cd build
```

2. Configure with CMake:
```bash
cmake ..
```

3. Build the project:
```bash
make
```

## Running the Vector Addition

The vector addition program demonstrates the performance difference between CPU and GPU implementations:

```bash
./vector_add
```

The program will:
- Generate random input vectors
- Perform vector addition on both CPU and GPU
- Compare results for correctness
- Display timing information and speedup

## Implementation Details

### CPU Implementation
- Sequential vector addition using a simple for loop
- Used as reference implementation for correctness verification

### GPU Implementation
- Parallel vector addition using CUDA
- Each thread handles one element
- Block size of 256 threads
- Automatic grid size calculation based on input size

### Performance Features
- CUDA events for accurate GPU timing
- Memory management optimization
- Error checking for CUDA operations
- Result verification with tolerance checking

## Testing

The implementation includes comprehensive testing:
- Various vector sizes (1 to 10M elements)
- Edge cases (zero vectors)
- Performance comparison
- Result verification

## Performance Considerations

The GPU implementation shows significant speedup over CPU for large vectors due to:
- Parallel processing of elements
- Efficient memory access patterns
- Optimized thread organization
- Minimal synchronization requirements

## Error Handling

The implementation includes robust error handling:
- CUDA operation error checking
- Memory allocation verification
- Result validation
- Graceful cleanup of resources

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NVIDIA CUDA Toolkit documentation
- CUDA Programming Guide
- CUDA Best Practices Guide
