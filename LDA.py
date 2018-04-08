import numpy as np

class LDAModel(object):

    def __init__(self, K, iter_times, top_Nwords):

        """
        K: number of topic
        alpha: prior
        beta: prior
        iter_times: iteration time
        top_words: top N words in topic_word distribution
        """

        self.K = K
        self.iter_times = iter_times
        self.top_Nwords = top_Nwords

    def fit(self, corpus):

        docs_count = 10
        distinct_words_count = 100

        # 2 distribution:
        # theta: Document-Topic distribution
        # phi: Topic-Word distribution
        theta = np.zeros((docs_count, self.K))
        phi = np.zeros((self.K, distinct_words_count), dtype="int")

        temp_p = np.zeros(self.K, dtype="double")
        doc_topic_count = np.zeros((docs_count, self.K), dtype="int")
        topic_wordcount = np.zeros(self.K, dtype="int")
        doc_wordcount = np.zeros(docs_count, dtype="int")



    def display(self):
        print(self.doc_topic_count)