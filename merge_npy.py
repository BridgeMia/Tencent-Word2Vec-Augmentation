from utils import load_array_multiprocess, save_list
import numpy as np

"""
Due to multi-process, we need a main function to avoid use multi-process in main process
"""

def main():
    
    separately_loaded_model = load_array_multiprocess()

    words = list(separately_loaded_model.keys())
    vecs = list(separately_loaded_model.values())

    save_list(words, r'data/array_raw/words.txt')
    np.save(r'data/array_raw/vectors.npy', vecs)


if __name__ == '__main__':
    main()
