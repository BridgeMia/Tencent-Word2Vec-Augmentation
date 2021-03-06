# Tencent-Word2Vec-Augmentation
Modification and Augmentation for Tencent AI Lab Embedding Corpus for Chinese Words and Phrases

## Data Source
 - [Tencent AI Lab Embedding Corpus for Chinese Words and Phrases](https://ai.tencent.com/ailab/nlp/embedding.html)
 - Paper: Yan Song, Shuming Shi, Jing Li, and Haisong Zhang. Directional Skip-Gram: Explicitly Distinguishing Left and Right Context for Word Embeddings. 
 NAACL 2018 (Short Paper) [[pdf](https://aclweb.org/anthology/N18-2028)]

## Preparation

Run files below **sequentially**: 

1. Split the raw txt file into `9` pieces, for multi-process loading

   run `split_raw_txt.py`

2. Transfer txt files to npy files for faster loading and less storage consumption

   run `txt2npy.py`

3. Merge split npy files into a single file

   run `merge_npy.py`

## Modification

### A dictionary model

The model is a  `dict`: `{word-str: vector-np.nparray}`

```python
key: 
    </s> 
vector: 
    [ 2.001e-03  2.211e-03 -1.915e-03 -1.639e-03  6.828e-04  1.511e-03
    ...
    -1.382e-03  8.769e-04  2.871e-04  8.950e-04 -5.908e-04  9.900e-05
    -8.430e-04 -5.631e-04]
```



- `load_txt_raw`
  - Load the original txt file
  - `load_mode`: default `generator` to save memory, or use `normal` for normal loader
  - `dtype`: vector array datatype, default `'float16'` is abundant

- `load_txt_multiprocess`
  - Load separate txt files in a multi-process way
  - `load_mode`: default `generator` to save memory, or use `normal` for normal loader
  - `dtype`: vector array datatype, default `'float16'` is abundant

- `load_array_multiprocess`
  - Load separate npy files(arrays) and corresponding txt files(words)
- `load_array_single`
  - Load a single npy file(arrays) and corresponding txt file(words)
  - 

### A trainable Gensim model