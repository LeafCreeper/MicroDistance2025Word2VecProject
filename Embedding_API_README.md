# Embedding 与 SequentialEmbedding API 使用教学（详细版）

本教程介绍如何在 HistWords 项目中使用 `Embedding` 和 `SequentialEmbedding` 两个核心类，加载词向量、获取词向量、计算相似度等，包含每个函数的参数与返回值说明，适合教学参考与项目开发使用。

---

## 1. 环境准备

确保已将 `histwords` 目录加入 `PYTHONPATH`，并已安装依赖。

$$$
import sys
sys.path.append('/root/workspace/MicroDistance-Word2Vec/histwords')
$$$

---

## 2. Embedding 类

### 2.1 加载词向量

$$$
from representations.embedding import Embedding
embedding = Embedding.load('/path/to/embedding_prefix')
$$$

- **参数**
  - `path` (str): 不带扩展名的文件前缀路径，后缀为 `-w.npy`、`-vocab.pkl`（如 `eng-all/1950`）
  - `normalize` (bool): 是否对向量行归一化，默认为 True
  - `add_context` (bool): 是否将词和上下文向量相加，仅适用于 SGNS 模型
- **返回值**
  - `Embedding` 实例

---

### 2.2 获取词向量

$$$
vector = embedding.represent('病毒')
$$$

- **参数**
  - `w` (str): 要获取的词
- **返回值**
  - `np.ndarray`: 维度为 `(dim,)` 的词向量。若词不在词表中则返回全零向量，并输出提示。

---

### 2.3 判断词是否在词表中

$$$
'病毒' in embedding  # True/False
$$$

- 使用 `__contains__` 魔法方法
- **返回值**
  - `bool`: True 表示词在词表中，False 表示 OOV

---

### 2.4 计算两个词的相似度（余弦相似度）

$$$
sim = embedding.similarity('病毒', '疾病')
$$$

- **参数**
  - `w1`, `w2` (str): 要比较的两个词
- **返回值**
  - `float`: 向量之间的余弦相似度（要求向量已归一化）

---

### 2.5 获取最相近的词

$$$
neighbors = embedding.closest('病毒', n=5)
for score, word in neighbors:
    print(word, score)
$$$

- **参数**
  - `w` (str): 中心词
  - `n` (int): 返回前 `n` 个最相近的词，默认 10
- **返回值**
  - `List[Tuple[float, str]]`: 向量相似度和词组成的列表，按相似度降序排列

---

### 2.6 获取与向量最相近的词

$$$
vec = embedding.represent('病毒')
embedding.vec_closest(vec, n=5)
$$$

- **参数**
  - `vec` (np.ndarray): 任意维度为 `dim` 的向量
  - `n` (int): 返回前 `n` 个相似词
- **返回值**
  - `List[Tuple[float, str]]`: 相似度与词的对

---

### 2.7 获取邻近词构成的子嵌入

$$$
subembed = embedding.get_neighbourhood_embed('病毒', n=10)
$$$

- **参数**
  - `w` (str): 中心词
  - `n` (int): 返回相似词数量
- **返回值**
  - `Embedding`: 子嵌入（包含 `w` 的最近邻）

---

### 2.8 获取指定词表构成的子嵌入（精简词表）

$$$
words = ['病毒', '疾病', '传染']
subembed = embedding.get_subembed(words)
$$$

- **参数**
  - `word_list` (List[str]): 要包含的词表，自动排除 OOV
- **返回值**
  - `Embedding`: 新的子嵌入对象

---

### 2.9 对齐词表（保序 + 补零）

$$$
aligned = embedding.reindex(['病毒', '感冒', '外星人'])
$$$

- **参数**
  - `word_list` (List[str]): 目标词表
- **返回值**
  - `Embedding`: 所有词顺序一致，OOV 词向量为全零

---

### 2.10 语义轴投影

$$$
embedding.get_projection(['男人', '他'], ['女人', '她'], '科学家')
$$$

- **参数**
  - `positive_words` (List[str]): 语义轴正向端（如男性）
  - `negative_words` (List[str]): 语义轴反向端（如女性）
  - `target_word` (str): 要测量的词
  - `verbose` (bool): 是否打印缺词信息
- **返回值**
  - `float` 或 `None`: 投影值 ∈ [-1, 1]，越接近正向端越大；如果无法计算则返回 None

---

## 3. SequentialEmbedding 类（历时语义分析）

用于加载多个年份的 `Embedding` 实例，追踪语义变化。

---

### 3.1 加载时间序列词向量

$$$
from representations.sequentialembedding import SequentialEmbedding
years = range(1950, 2000, 10)
semb = SequentialEmbedding.load('/path/to/embedding_dir', years)
$$$

- **参数**
  - `path` (str): 路径前缀，每年一个子文件夹（如 `eng-all/1950`）
  - `years` (Iterable[int]): 年份列表
- **返回值**
  - `SequentialEmbedding` 对象

---

### 3.2 获取某一年的嵌入对象

$$$
embed = semb.get_embed(1980)
$$$

- **参数**
  - `year` (int): 年份
- **返回值**
  - `Embedding`: 对应年份的嵌入对象

---

### 3.3 获取词在各年份的向量

$$$
for year in years:
    vec = semb.get_embed(year).represent('病毒')
    print(year, vec[:5])
$$$

- 无封装方法，推荐直接循环 `get_embed(year)`

---

### 3.4 获取词对随时间的相似度变化

$$$
semb.get_time_sims('病毒', '疾病')
$$$

- **返回值**
  - `Dict[int, float]`: 各年份的相似度字典

---

### 3.5 获取某词在所有年份的邻居集合

$$$
semb.get_seq_neighbour_set('病毒', n=5)
$$$

- **参数**
  - `word` (str): 中心词
  - `n` (int): 每年相邻词个数
- **返回值**
  - `Set[str]`: 所有年份中出现的邻居词合集（无重复）

---

### 3.6 获取每年的前 n 个最相似词

$$$
semb.get_seq_closest_by_year('病毒', n=5)
$$$

- **返回值**
  - `Dict[int, List[Tuple[float, str]]]`: 每年相似词列表

---

### 3.7 获取综合多年的最相似词

$$$
semb.get_seq_closest('病毒', start_year=1950, num_years=10, n=10)
$$$

- **返回值**
  - `List[str]`: 综合分数最高的前 n 个词

---

### 3.8 获取子词表的时间嵌入序列（可带随机词）

$$$
semb.get_word_subembeds('病毒', n=5, num_rand=10)
$$$

- **参数**
  - `word` (str): 中心词
  - `n` (int): 每年取 n 个相邻词
  - `num_rand` (int): 附加随机词数（默认 None）
- **返回值**
  - `SequentialEmbedding`: 仅包含子词表的向量序列

---

### 3.9 获取目标词在语义轴上的跨时间投影值

$$$
semb.get_projection_by_year(['男人', '他'], ['女人', '她'], '科学家')
$$$

- **返回值**
  - `Dict[int, float or None]`: 每年的语义投影分数

