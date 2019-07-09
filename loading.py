from functools import reduce
from multiprocessing import Pool

from tqdm import tqdm
import numpy as np

from constants import MODEL_SIZE
from utils import load_list


class Loader:
    def __init__(self):
        pass
    
    @staticmethod
    def load_txt_raw(fn=r'data/Tencent_AILab_ChineseEmbedding.txt', mode='generator', dtype='float16'):
        """
        Load the word2vec dict from raw txt file.
        Dict key is word and value is a np.ndarray of vector.

        :param fn: File name of the raw data file
        :param mode: Use generator or load the whole file
            generator: Use generator to save memory when load the txt file
            normal: Normal loading process
        :param dtype: dtype of the vector array,
                    default float16 is enough since the round of digit in the array is less than 5

        :return: dict of word2vec model
        """
    
        if mode == 'generator':
            # Use generator to save memory
            def line_generator():
                with open(fn, encoding='utf-8') as fin_:
                    for line_ in tqdm(fin_):
                        yield list(line_.split(' '))
                        del line_
                        # break
        
            ret = {x[0]: np.array([float(_) for _ in x[1:]], dtype=dtype) for x in line_generator()}
            try:
                ret.pop(str(MODEL_SIZE))
            except KeyError:
                pass
        
            return ret
    
        if mode == 'normal':
            ret = []
            with open(fn, encoding='utf-8') as fin:
                for line in fin:
                    ret.append(list(line.split(' ')))
                    del line
            ret = {x[0]: np.array([float(_) for _ in x[1:]], dtype=dtype) for x in ret[1:]}
            return ret
    
    @staticmethod
    def single_load_process(i, mode='generator', dtype='float16'):
        fn = r'data/txt_s/raw_txt_%d.txt' % i
        return Loader.load_txt_raw(fn, mode, dtype)
    
    @staticmethod
    def load_txt_multiprocess(mode='generator', dtype='float16'):
        pool = Pool(9)
        ret = []
        for i in range(1, 10):
            ret.append(pool.apply_async(Loader.single_load_process, args=(i, mode, dtype)))
        pool.close()
        pool.join()
    
        ret = reduce(lambda x1, x2: {**x1, **x2}, [x.get() for x in ret])
        return ret
    
    @staticmethod
    def load_array_single(key_fn: str, vec_fn: str):
        keys = load_list(key_fn)
        vecs = np.load(vec_fn)
        vecs = [np.hstack(v) for v in np.split(vecs, len(vecs))]
        return {k: v for k, v in zip(keys, vecs)}
    
    @staticmethod
    def single_load_array_process(i: int):
        key_fn_i = r'data/array_raw/words_%d.txt' % i
        vec_fn_i = r'data/array_raw/vectors_%d.npy' % i
        return Loader.load_array_single(key_fn_i, vec_fn_i)
    
    @staticmethod
    def load_array_multiprocess():
        pool = Pool(9)
        ret = []
        for i in range(1, 10):
            ret.append(pool.apply_async(Loader.single_load_array_process, args=[i]))
        pool.close()
        pool.join()
    
        ret = reduce(lambda x1, x2: {**x1, **x2}, [x.get() for x in ret])
        return ret