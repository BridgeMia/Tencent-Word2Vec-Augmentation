import datetime
from functools import reduce
import numpy as np
from multiprocessing import Pool, Queue, Process
from tqdm import tqdm
from constants import MODEL_SIZE


def load_txt_raw(fn=r'data/Tencent_AILab_ChineseEmbedding.txt', mode='generator', dtype='float16'):
    """
    Load the word2vec dict from raw txt file.
    Dict key is word and value is a np.ndarray of vector.

    :param fn: file name of the raw data file
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


def single_load_process(i, mode='generator', dtype='float16'):
    fn = r'data/txt_s/raw_txt_%d.txt' % i
    return load_txt_raw(fn, mode, dtype)


def load_txt_multiprocess(mode='generator', dtype='float16'):
    pool = Pool(9)
    ret = []
    for i in range(1, 10):
        ret.append(pool.apply_async(single_load_process, args=(i, mode, dtype)))
    pool.close()
    pool.join()

    ret = reduce(lambda x1, x2: {**x1, **x2}, [x.get() for x in ret])
    return ret


def load_array_raw(keys=r'data/array_raw/words.txt', vecs=r'data/array_raw/vectors.npy'):
    pass


def write_line_space(fout, lst):
    fout.write(' '.join([str(x) for x in lst]) + '\n')


def write_line_t(fout, lst):
    fout.write(' '.join([str(x) for x in lst]) + '\n')


def saveCSV(csv, fn):
    """
    Save a csv-like file into a .txt file
    :param csv: a list or 2d-array
    [[x11, x12, x13],
    [x21, x22, x23],
    [x31, x32, x33]]
    :param fn: save file path
    :return: None
    """
    with open(fn, 'w', encoding='utf-8') as fout:
        for x in csv:
            write_line_space(fout, x)


if __name__ == '__main__':
    t1 = datetime.datetime.now()

    model = load_txt_multiprocess()
    # model = load_txt_raw()
    print(len(model))

    for k, v in model.items():
        print(k, v)
        break

    t2 = datetime.datetime.now()
    print((t2 - t1).seconds)
