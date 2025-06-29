# Embedding 与 SequentialEmbedding API 使用教学

本教程介绍如何在 HistWords 项目中使用 `Embedding` 和 `SequentialEmbedding` 两个核心类，加载词向量、获取词向量、计算相似度等。

## 1. 环境准备

确保已将 `histwords` 目录加入 `PYTHONPATH`，并已安装依赖。

```python
import sys
sys.path.append('/root/workspace/MicroDistanc-Word2Vec/histwords')
```

## 2. Embedding 类

### 2.1 加载词向量

```python
from representations.embedding import Embedding
embedding = Embedding.load('/path/to/embedding_prefix')
```
- `/path/to/embedding_prefix` 为词向量文件前缀（不带扩展名），如 `Chinese_sgns_basic/1990`

### 2.2 获取词向量

```python
vector = embedding.represent('病毒')
```

### 2.3 判断词是否在词表中

```python
'病毒' in embedding  # True/False
```

### 2.4 计算两个词的相似度

```python
sim = embedding.similarity('病毒', '疾病')
```

### 2.5 获取最相近的词

```python
neighbors = embedding.closest('病毒', n=5)
for score, word in neighbors:
    print(word, score)
```

## 3. SequentialEmbedding 类

适用于加载多个时间点的历史词向量。

### 3.1 加载历史词向量序列

```python
from representations.sequentialembedding import SequentialEmbedding
years = range(1950, 2000, 10)
semb = SequentialEmbedding.load('/path/to/embedding_dir', years)
```

### 3.2 获取某一年份的 Embedding

```python
embed_2000 = semb.get_embed(1950)
```

### 3.3 获取某个词在各年份的向量

```python
for year in years:
    vec = semb.get_embed(year).represent('病毒')
    print(year, vec)
```

### 3.4 计算某两个词随时间的相似度变化

```python
time_sims = semb.get_time_sims('病毒', '疾病')
for year, sim in time_sims.items():
    print(year, sim)
```

### 3.5 获取某个词在所有年份的邻居集合

```python
neigh_set = semb.get_seq_neighbour_set('病毒', n=5)
print(neigh_set)
```

---

如需更多高级用法，请查阅源码或补充需求！
