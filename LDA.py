import numpy as np
import random

class LDAModel(object):

    def __init__(self, raw_text, K, iter_times, top_N_words, top_N_topics=1, alpha=0.1, beta=0.1):
        """
        raw_text: list of documents, where document is list of words
        K: number of topic
        alpha: prior
        beta: prior
        iter_times: iteration time
        top_words: top N words in topic_word distribution
        """

        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iter_times = iter_times
        self.top_N_words = top_N_words
        self.top_N_topics = top_N_topics

        # corpus is list(list) after word indexing
        # distict_word_count is dict(<word>, <count>)
        # distinct_word is list of distinct word, coresponding to word index in corpus

        self.corpus, self.distinct_words_count, self.distinct_word_list = self.preprocess(raw_text)
        self.distinct_wordcount = len(self.distinct_word_list)
        self.doc_count = len(self.corpus)
        self.wordcount_doc = [len(doc) for doc in self.corpus]
        self.p = np.zeros(self.K)

        # Logic check for model

        # top_N_words parameter cannot bigger than distinct_wordcount
        if self.top_N_words > self.distinct_wordcount:
            self.top_N_words = self.distinct_wordcount

        # top_N_topics parameter cannot bigger than total topic number K
        if self.top_N_topics > self.K:
            self.top_N_topics = self.K

    def preprocess(self, raw_text):
        # Word tokenizer and all the other stuffs
        tokenized = [text.split(" ") for text in raw_text]

        # Constructing distinct word list
        distinct_words_count = dict()
        for doc in tokenized:
            for word in doc:
                if distinct_words_count.__contains__(word):
                    distinct_words_count[word] += 1
                else:
                    distinct_words_count[word] = 1
        distinct_words_list = list(distinct_words_count.keys())

        # Word indexing using distinct_word_list
        corpus = [[distinct_words_list.index(word) for word in doc] for doc in tokenized]

        return corpus, distinct_words_count, distinct_words_list

    def topic_init(self):
        # Initialization of topic assignment matrix
        self.topic_assigned = np.array([[0 for word in list(range(self.wordcount_doc[doc]))] for doc in list(range(self.doc_count))])
        self.word_topic = np.zeros((self.distinct_wordcount, self.K), dtype="int")
        self.doc_topic = np.zeros((self.doc_count, self.K), dtype="int")
        self.wordcount_topic = np.zeros(self.K, dtype="int")
        for doc in range(len(self.wordcount_doc)):
            # "i" is the indexes of words in document
            for word in range(self.wordcount_doc[doc]):
                topic = random.randint(0, self.K-1)
                # assign generated topic to word
                self.topic_assigned[doc][word] = topic
                # increase topic count for specific word
                self.word_topic[self.corpus[doc][word]][topic] += 1
                # increase topic count for specific document
                self.doc_topic[doc][topic] += 1
                # increase word count for specific topic
                self.wordcount_topic[topic] += 1

        # Initialize theta and phi
        self.theta = np.array([[0.0 for topic_count in list(range(self.K))] for doc_count in list(range(self.doc_count))])
        self.phi = np.array([[0.0 for word_count in list(range(self.distinct_wordcount))] for topic_count in list(range(self.K))])

    def sampler(self, doc_index, word_index):
        # sample topic for a specific word from the rest of word
        topic = self.topic_assigned[doc_index][word_index]
        # word here is actually a word index in the distinct_word_list
        word = self.corpus[doc_index][word_index]
        
        # decrease counts for specific word data
        self.word_topic[word][topic] -= 1   # word-topic distribution
        self.doc_topic[doc_index][topic] -= 1   # document-topic distribution
        self.wordcount_topic[topic] -= 1    # wordcount-topic distribution
        self.wordcount_doc[doc_index] -= 1 # wordcount-document distribution

        distinct_wordcount = self.distinct_wordcount-1
        w_beta = distinct_wordcount * self.beta
        k_alpha = self.K * self.alpha
        self.p = (self.word_topic[word] + self.beta)/(self.wordcount_topic + w_beta) * \
                 (self.doc_topic[doc_index] + self.alpha)/(self.wordcount_doc[doc_index] + k_alpha)

        for k in range(1, self.K):
            self.p[k] += self.p[k-1]

        u = random.uniform(0, self.p[self.K-1])
        for topic in range(self.K):
            if self.p[topic]>u:
                break

        self.word_topic[word][topic] += 1  # word-topic distribution
        self.doc_topic[doc_index][topic] += 1  # document-topic distribution
        self.wordcount_topic[topic] += 1  # wordcount-topic distribution
        self.wordcount_doc[doc_index] += 1  # wordcount-document distribution

        return topic

    def theta_infer(self):
        for d in range(self.doc_count):
            # theta[d]: topic distribution for document[d]
            self.theta[d] = (self.doc_topic[d] + self.alpha)/(self.wordcount_doc[d] + self.K * self.alpha)

    def phi_infer(self):
        for k in range(self.K):
            # phi[k]: word distribution for topic[k]
            self.phi[k] = (self.word_topic.transpose()[k] + self.beta)/(self.wordcount_topic[k] + self.distinct_wordcount * self.beta)

    def dist_interpret(self):
        self.doc_topic_outcome = list()
        self.topic_word_outcome = list()

        for d in range(self.doc_count):
            temp_map = dict()
            for k in range(self.K):
                temp_map["Topic "+str(k)] = self.theta[d][k]
            self.doc_topic_outcome.append(sorted(temp_map.items(), key=lambda d: d[1], reverse=True))

        for k in range(self.K):
            temp_map = dict()
            for w in range(self.distinct_wordcount):
                temp_map[self.distinct_word_list[w]] = self.phi[k][w]
            self.topic_word_outcome.append(sorted(temp_map.items(), key=lambda d: d[1], reverse=True))

    def fit(self):
        print("[Console Log]: Iteration set {} times.".format(self.iter_times))
        print("[Console Log]: Iteration Start...")
        for iter in range(self.iter_times):
            # for every iteration
            # update topic assignment for every word in every document
            for doc in range(self.doc_count):
                for word in range(self.wordcount_doc[doc]):
                    topic = self.sampler(doc_index=doc, word_index=word)
                    self.topic_assigned[doc][word] = topic

        print("[Console Log]: Iteration finished.")
        print("[Console Log]: Inferring theta...")
        self.theta_infer()
        print("[Console Log]: Inferring phi...")
        self.phi_infer()

        print("[Console Log]: Interpreting theta and phi...")
        self.dist_interpret()

    def display_var(self):
        print("Topic assignments for words: \n{}\n Topic distribution for words: \n{}\n Topic distribution for docutments: \n{}\n Word count for Topics: \n{}\n"\
              .format(self.topic_assigned, self.word_topic, self.doc_topic, self.wordcount_topic))

    def display_result(self):
        for d in range(self.doc_count):
            print("========Document [{}]========".format(d))
            for topN_k in range(self.top_N_topics):
                topic_tuple = self.doc_topic_outcome[d][topN_k]
                topic_index = int(topic_tuple[0].split(" ")[1])
                print("{} [{}]: ".format(topic_tuple[0], topic_tuple[1]), end="")
                # topic_index is the specific topic in doc-topic sorted list
                topic_words = self.topic_word_outcome[topic_index]

                word_list = ""
                for topN_w in range(self.top_N_words):
                    word_list += (str(topic_words[topN_w][0]) + "[" + str(topic_words[topN_w][1]) + "] ")
                print(word_list)
