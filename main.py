from gensim.models import KeyedVectors
import argparse
import pprint
import numpy as np
from collections import Counter

pp = pprint.PrettyPrinter()
parser = argparse.ArgumentParser()
parser.add_argument('--inputfile', type=str, required=True)



def compute_cosine_similarity(vector1, vector2):
    '''Computes the cosine similarity of the two input vectors.
      Inputs:
        vector1: A nx1 numpy array
        vector2: A nx1 numpy array

      Returns:
        A scalar similarity value.
      '''

    # YOUR CODE HERE
    cos = vector1.dot(vector2) / (np.linalg.norm(vector1, ord=2) * np.linalg.norm(vector2, ord=2))
    if np.isnan(cos):
        return 0.500    # arbitrarily low semantic similarity
    else:
        return cos


def simple_baseline(input, output):
    write_str = []
    sims = []
    s_min = 1
    s_max = 0

    for line in input:
        line_components = line.split("\t")
        sent1 = line_components[0]
        sent2 = line_components[1]  
        words_in_sent1 = sent1.split()
        words_in_sent2 = sent2.split()

        # construct temp. tf-idf
        # Note: tf_idf is not a set so we can index into it
        tf_idf = []
        tf_idf1 = Counter()
        tf_idf2 = Counter()
        for word in words_in_sent1:
            if word not in tf_idf:
                tf_idf.append(word)
            tf_idf1[word] += 1
        for word in words_in_sent2:
            if word not in tf_idf:
                tf_idf.append(word)
            tf_idf2[word] += 1

        n = len(tf_idf)
        v1 = np.zeros(n)
        v2 = np.zeros(n)
        for i in range(0, n):
            v1[i] = tf_idf1[tf_idf[i]]
            v2[i] = tf_idf2[tf_idf[i]]

        sim = compute_cosine_similarity(v1, v2)

        write_str.append(sent1 + "\t" + sent2 + "\t")
        sims.append(sim)

        s_max = max(s_max, sim)
        s_min = min(s_min, sim)

    sims_scaled = [5*(i - s_min)/(s_max - s_min) for i in sims]
    for i in range(0,len(write_str)):
        output.write(write_str[i] + str(sims_scaled[i]) + "\n")

    pass


def predict_semantic_sim(input, vecs, output):
    write_str = []
    sims = []
    s_min = 1
    s_max = 0

    for line in input:
        line_components = line.split("\t")
        sent1 = line_components[0]
        sent2 = line_components[1]
        words_in_sent1 = sent1.split()
        words_in_sent2 = sent2.split()
        v1 = np.zeros(vecs["hi"].shape)
        for word in words_in_sent1:
            if word in vecs:
                v1 = v1 + np.asarray(vecs[word])

        v2 = np.zeros(vecs["hi"].shape)
        for word in words_in_sent2:
            if word in vecs:
                v2 = v2 + np.asarray(vecs[word])

        sim = compute_cosine_similarity(v1, v2)

        write_str.append(sent1 + "\t" + sent2 + "\t")
        sims.append(sim)

        s_max = max(s_max, sim)
        s_min = min(s_min, sim)

    sims_scaled = [5*(i - s_min)/(s_max - s_min) for i in sims]
    for i in range(0,len(write_str)):
        output.write(write_str[i] + str(sims_scaled[i]) + "\n")

    pass

def sub_cost(char1, char2):
    if char1 == char2:
        return 0
    else:
        return 2

def edit_distance(str1, str2):
    '''Computes the minimum edit distance between the two strings.

    Use a cost of 1 for all operations.

    See Section 2.4 in Jurafsky and Martin for algorithm details.
    Do NOT use recursion.

    Returns:
    An integer representing the string edit distance
    between str1 and str2
    '''
    n = len(str1)
    m = len(str2)
    D = [[0 for i in range(m+1)] for j in range(n+1)]
    D[0][0] = 0
    for i in range(1,n+1):
        D[i][0] = D[i-1][0] + 1
    for j in range(1,m+1):
        D[0][j] = D[0][j-1] + 1
    for i in range(1,n+1):
        for j in range(1,m+1):
            D[i][j] = min(D[i-1][j]+1, D[i-1][j-1]+sub_cost(str1[i-1],str2[j-1]), D[i][j-1]+1)
    return D[n][m]

if __name__ == '__main__':
    args = parser.parse_args()
    pp.pprint(args)

    # Load the training data
    vecfile = 'GoogleNews-vectors-negative300.bin'
    vecs = KeyedVectors.load_word2vec_format(vecfile, binary=True)

    with open(args.inputfile, 'r') as inputfile:
        input = inputfile.readlines()

    output_simple = open("pred_simple.txt", "w")
    output_w2v = open("pred_ex1.txt", "w")

    # predict_semantic_sim(input, vecs, output_w2v)


    # Close files
    output_simple.close()
    output_w2v.close()
    input.close()