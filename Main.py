from LDA import LDAModel

if __name__ == "__main__":
    alpha = 1
    beta = 1
    model = LDAModel(K=10, alpha=alpha, beta=beta, iter_times=1000, top_Nwords=5)
