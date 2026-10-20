import numpy as np

def softmax(x: list) -> np.ndarray:
    """
    Returns stable softmax probabilities as a NumPy array matching the shape of x.
    """
    # Write code here
    x = np.array(x)
    max_x = np.max(x,axis = -1, keepdims = True)
    e = np.exp(x-max_x)
    s = np.sum(e,axis = -1, keepdims = True)
    return e/s