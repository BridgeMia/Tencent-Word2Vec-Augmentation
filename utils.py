import numpy as np
# from threading import Thread
from tqdm import tqdm


def load_txt_raw(fn=r'data/Tencent_AILab_ChineseEmbedding.txt'):
    ret = []
    with open(fn) as fin:
        for line in tqdm(fin):
            ret.append(line)

    ret = {x[0]: np.array([float(_) for _ in x[1:]]) for x in ret[1:]}
    return ret


def load_array_raw(keys=r'data/array_raw/words.txt', vecs=r'data/array_raw/vectors.npy'):
    pass


if __name__ == '__main__':
    for k,v in load_txt_raw().items():
        print(k, v)
        break
