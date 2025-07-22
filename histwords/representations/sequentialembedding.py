import collections
import random

from representations.embedding import Embedding, SVDEmbedding
import numpy as np

class SequentialEmbedding:
    def __init__(self, year_embeds, **kwargs):
        self.embeds = year_embeds
 
    @classmethod
    def load(cls, path, years, **kwargs):
        embeds = collections.OrderedDict()
        for year in years:
            embeds[year] = Embedding.load(path + "/" + str(year), **kwargs)
        return SequentialEmbedding(embeds)

    def get_embed(self, year) -> Embedding:
        return self.embeds[year]

    def get_subembeds(self, words, normalize=True):
        embeds = collections.OrderedDict()
        for year, embed in list(self.embeds.items()):
            embeds[year] = embed.get_subembed(words, normalize=normalize)
        return SequentialEmbedding(embeds)

    def get_time_sims(self, word1, word2):
       time_sims = collections.OrderedDict()
       for year, embed in list(self.embeds.items()):
           time_sims[year] = embed.similarity(word1, word2)
       return time_sims

    def get_seq_neighbour_set(self, word, n=5):
        neighbour_set = set([])
        for embed in list(self.embeds.values()):
            closest = embed.closest(word, n=n)
            for _, neighbour in closest:
                neighbour_set.add(neighbour)
        return neighbour_set

    def get_seq_closest(self, word, start_year, num_years=10, n=10):
        closest = collections.defaultdict(float)
        for year in range(start_year, start_year + num_years):
            embed = self.embeds[year]
            year_closest = embed.closest(word, n=n*10)
            for score, neigh in year_closest:
                closest[neigh] += score
        return sorted(closest, key = lambda word : closest[word], reverse=True)[0:n]

    def get_seq_closest_by_year(self, word, n=10):
        """
        获取每个年份中与目标词最相近的前 n 个词。
        返回格式：{year: [(score, word), ...], ...}
        """
        result = {}
        for year, embed in self.embeds.items():
            neighbors = embed.closest(word, n=n)
            result[year] = neighbors
        return result


    def get_word_subembeds(self, word, n=3, num_rand=None, word_list=None):
        if word_list == None:
            word_set = self.get_seq_neighbour_set(word, n=n)
            if num_rand != None:
                # Python3: dict.values() returns a view, need to convert to list
                last_embed = list(self.embeds.values())[-1]
                word_set = word_set.union(set(random.sample(last_embed.iw, num_rand)))
            word_list = list(word_set)
        year_subembeds = collections.OrderedDict()
        for year,embed in list(self.embeds.items()):
            year_subembeds[year] = embed.get_subembed(word_list)
        return SequentialEmbedding(year_subembeds)
    
    def get_projection_by_year(self, positive_words, negative_words, target_word):
        """
        计算 target_word 在语义轴（由正反义词组成）上的投影随时间的变化。

        返回:
            dict: {year: projection_score (float or None)}

        如果某些词不在词表中，会跳过这些词，同时打印调试信息。
        """
        projection_scores = {}
        for year, embed in self.embeds.items():
            # 检查词是否存在
            missing_pos = [w for w in positive_words if w not in embed]
            missing_neg = [w for w in negative_words if w not in embed]
            target_missing = target_word not in embed

            if missing_pos or missing_neg or target_missing:
                print(f"[提示] {year} 年词表中缺失词汇："
                      f"{'正向词: ' + ', '.join(missing_pos) if missing_pos else ''} "
                      f"{'反向词: ' + ', '.join(missing_neg) if missing_neg else ''} "
                      f"{'目标词: ' + target_word if target_missing else ''}")

            if target_missing:
                projection_scores[year] = None
                continue

            valid_pos = [embed.represent(w) for w in positive_words if w in embed]
            valid_neg = [embed.represent(w) for w in negative_words if w in embed]

            if not valid_pos or not valid_neg:
                projection_scores[year] = None
                continue

            pos_vec = np.mean(valid_pos, axis=0)
            neg_vec = np.mean(valid_neg, axis=0)
            axis_vec = pos_vec - neg_vec

            target_vec = embed.represent(target_word)
            projection = np.dot(target_vec, axis_vec) / (
                np.linalg.norm(target_vec) * np.linalg.norm(axis_vec))
            projection_scores[year] = float(projection)

        return projection_scores

    


class SequentialSVDEmbedding(SequentialEmbedding):

    def __init__(self, path, years, **kwargs):
        self.embeds = collections.OrderedDict()
        for year in years:
            self.embeds[year] = SVDEmbedding(path + "/" + str(year), **kwargs)


