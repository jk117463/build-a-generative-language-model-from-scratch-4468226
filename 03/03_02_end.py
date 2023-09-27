import os
import numpy
import pickle
from numpy import dot
from numpy.linalg import norm


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(CURRENT_DIR + "/word_to_vector_trsf.pkl", "rb") as pk:
    word_to_vector = pickle.load(pk)

def cosine_similarity(vec_a, vec_b):
    numerator = sum([vec_a[i] * vec_b[i] for i in range(len(vec_a))])
    denominator = (norm(vec_a) * norm(vec_b))
    return numerator / denominator

def similar_words(word="tree", top_k =10):
    return sorted(
        word_to_vector.keys(),
        key=lambda x: -cosine_similarity(word_to_vector[x], word_to_vector[word])
    )

print(cosine_similarity(word_to_vector["plant"], word_to_vector["grow"]))
print(cosine_similarity(word_to_vector["minute"], word_to_vector["plant"]))
print(cosine_similarity(word_to_vector["plant"], word_to_vector["tree"]))

similar_words_list = similar_words("tree", top_k=10)
print(similar_words_list)