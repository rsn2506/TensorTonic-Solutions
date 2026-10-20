import numpy as np

def kl_divergence(p: list, q: list, eps: float = 1e-12) -> float:
    """
    Returns the divergence as a float.
    """
    # Write code here
    p = np.array(p)
    q = np.array(q)
    pp = np.where(p==0,0,p*np.log(p))
    qq = np.where(q==0,0,p*np.log(q))
    s = np.sum(pp-qq)
    return float(s)