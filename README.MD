# RLtools Example: Implementing a Custom Environment

## UNIX (Linux / macOS)
```
git clone https://github.com/rl-tools/example
cd example
git submodule update --init external/rl_tools
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
./src/my_pendulum
```



## Windows
On windows you should use `cmake --build . --config Release` to build the executable. If you have [Intel MKL/oneMKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html) installed you should use the `Command Prompt` (not PowerShell) and initialize the MKL environment variables before running the commands:

```
"C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
git clone https://github.com/rl-tools/example
cd example
git submodule update --init external/rl_tools
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
Release\my_pendulum.exe
```

## BLAS Backend & Optimizations

This example should work out of the box, but it is relatively slow if you do not activate one of the BLAS backends. Depending on your platform uncomment the `set(RL_TOOLS_BACKEND_ENABLE_XXX ON)` line in `CMakeLists.txt` and re-run the previous cmake config and build steps to dispatch to the optimized BLAS routines:
This should be substantially faster. To make it even fast you can disable the evaluation episodes (by enabling the `BENCHMARK` flag in the `CMakeLists.txt`). Furthermore, you can play with options like `-Ofast` and `-march=native` depending on your CPU architecture. For reference, this takes about 600ms on an `i9-10885H` and about 300ms on an `Apple M3`.
