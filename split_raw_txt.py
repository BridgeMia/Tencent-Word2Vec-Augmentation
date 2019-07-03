"""
Split raw txt data file into pieces in order to use Threading to accelerate
"""
from constants import FILE_HEAD, MODEL_SIZE
import os


# Use generator to save memory
# Load raw txt file
def line_generator():
    with open(r'data/Tencent_AILab_ChineseEmbedding.txt', encoding='utf-8') as fin:
        for _line in fin:
            yield _line
            del _line


def write_str_line(fout, lst):
    for line in lst:
        fout.write(line)


# Generate the word2vec list and split into 8 pieces
# Separately stored data will be in the same form of raw data
if not os.path.exists(r'data/txt_s'):
    os.mkdir(r'data/txt_s')

counter = 0
PIECE_SIZE = int(MODEL_SIZE / 8)
current_ret = []
file_num = 0
for line in line_generator():
    print(counter, '/', MODEL_SIZE)
    if line == FILE_HEAD:
        continue
    current_ret.append(line)
    counter += 1
    if int(counter / PIECE_SIZE) > file_num:
        file_num += 1
        print('saving ...')
        with open(r'data/txt_s/raw_txt_%d.txt' % file_num, 'w', encoding='utf-8') as fout:
            write_str_line(fout, current_ret)
        current_ret = []
        print('file saved')
    if counter == MODEL_SIZE:
        file_num += 1
        with open(r'data/txt_s/raw_txt_%d.txt' % file_num, 'w', encoding='utf-8') as fout:
            write_str_line(fout, current_ret)


print('Raw file slit into 8 pieces and saved in /data/txt_s/ dir')

