import numpy as np
from utils import load_txt_raw

"""
Script: Transfer .txt raw data to .npy data

"""
# Size: 8824330
# Dim: 200

# dict
# word(str): vector(np.ndarray)
w2v_model = load_txt_raw()
