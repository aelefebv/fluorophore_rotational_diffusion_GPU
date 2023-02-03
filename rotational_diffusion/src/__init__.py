try:
    import cupy as np
    print("[INFO] Running on GPU.")
except ModuleNotFoundError:
    import numpy as np
    print("[INFO] Running on CPU.")

