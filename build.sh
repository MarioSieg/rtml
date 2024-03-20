# Run cmake and build dll
# Usage: ./build.sh

cmake -S . -B bin
cmake --build bin --config Release
