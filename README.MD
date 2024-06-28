# RLtools Example: Implementing a Custom Environment
This example implements a full pendulum environment including a visualization in `191` lines of code (`115` for the environment + `76` for the UI). You can find the implementation in [include/my_pendulum](./include/my_pendulum)
## UNIX (Linux / macOS)
```
git clone https://github.com/rl-tools/example
cd example
git submodule update --init external/rl_tools
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
./my_pendulum
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
This should be substantially faster. To make it even fast you can disable the evaluation episodes (by enabling the `BENCHMARK` flag in the `CMakeLists.txt`). Furthermore, you can play with options like `-Ofast` and `-march=native` depending on your CPU architecture. For reference, this takes about 600ms on an `i9-10885H` (on Linux, on Windows around 1.5s for some reason) and about 300ms on an `Apple M3`.


## Experiment Tracking

<div style="text-align: center;">
<img src="https://raw.githubusercontent.com/rl-tools/media/master/experiment-tracking-screenshot.png" alt="Experiment Tracking Screenshot" width="500">
</div>

This example also includes the automatic experiment tracking available through the [Loop Interface](https://docs.rl.tools/07-The%20Loop%20Interface.html). More infos about the experiment tracking can be found in the docs: [Experiment Tracking](https://docs.rl.tools/10-Experiment%20Tracking.html). To enable the visualization of trajectories for a custom environment three functions are required (these are defined in `operations_cpu.h` due to their dependency on `std::string`):
1. [Parameters => JSON](https://github.com/rl-tools/example/blob/39acaa5b5402eacf5c2cab7b2e96db71f2ea110f/include/my_pendulum/operations_cpu.h#L3): Self-explanatory
2. [State => JSON](https://github.com/rl-tools/example/blob/39acaa5b5402eacf5c2cab7b2e96db71f2ea110f/include/my_pendulum/operations_cpu.h#L8): Self-explanatory
3. [UI Render function string](https://github.com/rl-tools/example/blob/39acaa5b5402eacf5c2cab7b2e96db71f2ea110f/include/my_pendulum/operations_cpu.h#L16): This function uses the HTML5 Canvas rendering API and can be easily created using [https://studio.rl.tools](https://studio.rl.tools). Nnote that due to the wide spread use of the HTML5 Canvas drawing API, also ChatGPT is really good at creating render functions for different environments if you give it an example like the ones provided on [https://studio.rl.tools](https://studio.rl.tools).

The experiment tracking and save-trajectories step will periodically record trajectories and store them as `.json` files. After/while running the training you can run `./serve.sh` which should start a local webserver on [http://localhost:8080](http://localhost:8080) where you can see the recorded trajectories based on the render function you provided.