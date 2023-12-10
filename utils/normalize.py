import numpy as np
from config import Config as con

def nor(data,vmin,vmax):
    res=data-vmin
    res=res/(vmax-vmin)
    return res
