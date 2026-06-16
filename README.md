## Dependencies
- RAJA
- Umpire

## Build

```bash
# Create and enter the build directory
mkdir build
cd build

# Configure the project
# Replace 'HIP' or 'CUDA' with the appropriate backend for your hardware
cmake -DBACKEND=HIP ..  # Or -DBACKEND=CUDA
export CHECKERS_ANALYZE_DEBUG_SYNC=1

# Compile the project
# The -j flag uses all available CPU cores for faster compilation
make -j
```

## Usage
```bash
./examples/script_llama_zero3.sh
