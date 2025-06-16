import json
import subprocess

def benchmark_headless(compilers = ["nvcc"], flags=["-O0"], grid_size = [128]):
    with open("data.json", "w") as file:
        data = []
        for size in grid_size:
            for compiler in compilers:
                for flag in flags:
                    cmd = ["make", f"NVCC={compiler}", f"NVFLAGS={flag} -DGRID_SIZE={size}"]
                    subprocess.run(cmd)

                    run = ["./headless"]
                    res = subprocess.run(run, capture_output=True)
                    cells_ms, time = res.stdout.decode().split()
                    data.append({"Compiler": compiler, 
                                 "Flags": flag, 
                                 "cellms": cells_ms, 
                                 "grid_size": size,
                                 "time": time})

                    clean = ["make", "clean"]
                    subprocess.run(clean)
        json.dump(data, file, indent=1)

compilers = ["nvcc"]
flags = ['-O3 -Xcompiler "-Wall -Wextra -Wno-unused-parameter -O3 -ffast-math -march=native -ftree-vectorize"']
grid_size = [128, 256]
benchmark_headless(compilers, flags, grid_size)
