import os

import numpy as np
from utils import save_list

"""
Script: Transfer .txt raw data to .npy data

"""
# Size: 8824330
# Dim: 200

# dict
# word(str): vector(np.ndarray)


txt_dirs = r'data/txt_s/'
i = 1
for file in os.listdir(txt_dirs):
	fn = os.path.join(txt_dirs, file)
	p_model = load_txt_raw(fn)
	p_words = []
	p_vectors = []
	
	for k, v in p_model.items():
		p_words.append(k)
		p_vectors.append(v)
	save_list(p_words, r'data/array_raw/words_%d.txt' % i)
	p_vectors = np.array(p_vectors)
	np.save(r'data/array_raw/vectors_%d.npy' % i,p_vectors)
	i += 1
