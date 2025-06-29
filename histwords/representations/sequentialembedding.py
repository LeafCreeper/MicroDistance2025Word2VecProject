import collections
import random

from representations.embedding import Embedding, SVDEmbedding

class SequentialEmbedding:
    def __init__(self, year_embeds, **kwargs):
        self.embeds = year_embeds
 
    @classmethod
    def load(cls, path, years, **kwargs):
        embeds = collections.OrderedDict()
        for year in years:
            embeds[year] = Embedding.load(path + "/" + str(year), **kwargs)
        return SequentialEmbedding(embeds)

    def get_embed(self, year):
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

    def get_seq_neighbour_set(self, word, n=3):
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


class SequentialSVDEmbedding(SequentialEmbedding):

    def __init__(self, path, years, **kwargs):
        self.embeds = collections.OrderedDict()
        for year in years:
            self.embeds[year] = SVDEmbedding(path + "/" + str(year), **kwargs)


