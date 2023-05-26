## GPU-agnostic
# Note, if number of molecules is small, then the GPU will be slower than the CPU.
# This is because the GPU has to copy the data from the CPU to the GPU, and then copy it back.
# If the number of molecules is large, then the GPU will be faster than the CPU.
# This is because the GPU can do many calculations in parallel.
# Also note that if the simulation is very large, the GPU may run out of memory where the CPU wouldn't,
#  but at that point the simulation would take a very long time anyway.

try:
    import cupy as np
    print("[INFO] Running on GPU.")
except ModuleNotFoundError:
    import numpy as np
    print("[INFO] Running on CPU.")
