import heapq
import numpy as np
from sklearn import preprocessing

from ioutils import load_pickle
#, lines

class Embedding:
    """
    Base class for all embeddings. SGNS can be directly instantiated with it.
    """

    def __init__(self, vecs, vocab, normalize=True, **kwargs):
        self.m = vecs
        self.dim = self.m.shape[1]
        self.iw = vocab
        self.wi = {w:i for i,w in enumerate(self.iw)}
        if normalize:
            self.normalize()

    def __getitem__(self, key):
        if self.oov(key):
            raise KeyError(key)
        else:
            return self.represent(key)

    def __iter__(self):
        return self.iw.__iter__()

    def __contains__(self, key):
        return not self.oov(key)

    @classmethod
    def load(cls, path, normalize=True, add_context=False, **kwargs):
        mat = np.load(path + "-w.npy", mmap_mode="c")
        if add_context:
            mat += np.load(path + "-c.npy", mmap_mode="c")
        iw = load_pickle(path + "-vocab.pkl")
        return cls(mat, iw, normalize) 

    def get_subembed(self, word_list, **kwargs):
        word_list = [word for word in word_list if not self.oov(word)]
        keep_indices = [self.wi[word] for word in word_list]
        return Embedding(self.m[keep_indices, :], word_list, normalize=False)

    def reindex(self, word_list, **kwargs):
        new_mat = np.empty((len(word_list), self.m.shape[1]))
        valid_words = set(self.iw)
        for i, word in enumerate(word_list):
            if word in valid_words:
                new_mat[i, :] = self.represent(word)
            else:
                new_mat[i, :] = 0 
        return Embedding(new_mat, word_list, normalize=False)

    def get_neighbourhood_embed(self, w, n=1000):
        neighbours = self.closest(w, n=n)
        keep_indices = [self.wi[neighbour] for _, neighbour in neighbours] 
        new_mat = self.m[keep_indices, :]
        return Embedding(new_mat, [neighbour for _, neighbour in neighbours]) 

    def normalize(self):
        preprocessing.normalize(self.m, copy=False)

    def oov(self, w):
        return not (w in self.wi)

    def represent(self, w):
        if w in self.wi:
            return self.m[self.wi[w], :]
        else:
            print(f"[OOV] 词{w}不在词表中")
            return np.zeros(self.dim)

    def similarity(self, w1:str, w2:str):
        """
        Assumes the vectors have been normalized.
        """
        sim = self.represent(w1).dot(self.represent(w2))
        return sim

    def closest(self, w, n=10):
        """
        Assumes the vectors have been normalized.
        """
        scores = self.m.dot(self.represent(w))
        return heapq.nlargest(n, list(zip(scores, self.iw)))
    
    def vec_closest(self, vec, n=10):
        scores = self.m.dot(vec)  # self.m 是词向量矩阵，行为向量
        return heapq.nlargest(n, zip(scores, self.iw))
    
    def get_projection(self, positive_words, negative_words, target_word, verbose=True):
        """
        计算目标词在由正反义词构成的语义轴上的投影分数。

        参数:
            positive_words (list[str]): 正向词列表，如 ["man", "he"]
            negative_words (list[str]): 反向词列表，如 ["woman", "she"]
            target_word (str): 需要投影的词
            verbose (bool): 是否打印缺失词信息

        返回:
            float or None: 投影分数（[-1, 1]区间），若无法计算则返回 None
        """
        missing_pos = [w for w in positive_words if w not in self]
        missing_neg = [w for w in negative_words if w not in self]
        target_missing = target_word not in self

        if verbose and (missing_pos or missing_neg or target_missing):
            print(f"[提示] 缺失词："
                  f"{'正向词: ' + ', '.join(missing_pos) if missing_pos else ''} "
                  f"{'反向词: ' + ', '.join(missing_neg) if missing_neg else ''} "
                  f"{'目标词: ' + target_word if target_missing else ''}")

        if target_missing:
            print("目标词不在词表中，无法计算")
            return None

        valid_pos = [self.represent(w) for w in positive_words if w in self]
        valid_neg = [self.represent(w) for w in negative_words if w in self]

        if not valid_pos or not valid_neg:
            return None

        pos_vec = np.mean(valid_pos, axis=0)
        neg_vec = np.mean(valid_neg, axis=0)
        axis_vec = pos_vec - neg_vec

        target_vec = self.represent(target_word)
        projection = np.dot(target_vec, axis_vec) / (
            np.linalg.norm(target_vec) * np.linalg.norm(axis_vec))

        return float(projection)



class SVDEmbedding(Embedding):
    """
    SVD embeddings.
    Enables controlling the weighted exponent of the eigenvalue matrix (eig).
    Context embeddings can be created with "transpose".
    """
    
    def __init__(self, path, normalize=True, eig=0.0, **kwargs):
        ut = np.load(path + '-u.npy', mmap_mode="c")
        s = np.load(path + '-s.npy', mmap_mode="c")
        vocabfile = path + '-vocab.pkl'
        self.iw = load_pickle(vocabfile)
        self.wi = {w:i for i, w in enumerate(self.iw)}
 
        if eig == 0.0:
            self.m = ut
        elif eig == 1.0:
            self.m = s * ut
        else:
            self.m = np.power(s, eig) * ut

        self.dim = self.m.shape[1]

        if normalize:
            self.normalize()

class GigaEmbedding(Embedding):
    def __init__(self, path, words=[], dim=300, normalize=True, **kwargs):
        seen = []
        vs = {}
        for line in open(path, encoding='utf-8'):
            split = line.split()
            w = split[0]
            if words == [] or w in words:
                if len(split) != dim+1:
                    continue
                seen.append(w)
                vs[w] = np.array(list(map(float, split[1:])), dtype='float32')
        self.iw = seen
        self.wi = {w:i for i,w in enumerate(self.iw)}
        self.m = np.vstack([vs[w] for w in self.iw])
        if normalize:
            self.normalize()


