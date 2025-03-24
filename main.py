import json
import subprocess

def benchmark_headless(compilers, flags):
    with open("data.json", "w") as file:
        data = []
        for compiler in compilers:
            for flag in flags:
                cmd = ["make", f"CC={compiler}", f"CFLAGS={flag}"]
                subprocess.run(cmd)

                run = ["./headless"]
                res = subprocess.run(run, capture_output=True)
                cells_ms = float(res.stdout.decode().strip())
                data.append({"Compiler": compiler, "Flags": flag, "cellms": cells_ms})

                clean = ["make", "clean"]
                subprocess.run(clean)
        json.dump(data, file, indent=1)

compilers = ["gcc", "clang"]
flags = ["-O0", "-O1"]
benchmark_headless(compilers, flags)
