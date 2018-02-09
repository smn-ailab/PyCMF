import numpy as np

def _print_topic_terms_from_matrix(term_topic_matrix, idx_to_word,
                                   topn_words=10, n_topics=100):
    for i, topic in enumerate(term_topic_matrix.T[:n_topics]):
        top_terms = idx_to_word[topic.argsort()[-10:]]
        print("Topic {}: {}".format(i + 1, ",".join([str(x) for x in top_terms])))

        
def _print_topic_terms_with_importances_from_matrices(term_topic_matrix,
                                                      cv_topic_matrix, idx_to_word,
                                                      topn_words=10, n_topics=100):
    for i, (topic, weights) in enumerate(zip(term_topic_matrix.T[:n_topics], cv_topic_matrix.T[:n_topics])):
        top_terms = idx_to_word[topic.argsort()[-10:]]
        weight_str = ",".join(["{:.3f}".format(x) for x in weights])
        print("Topic {} [{}]: {}".format(i + 1, weight_str, ",".join([str(x) for x in top_terms])))
