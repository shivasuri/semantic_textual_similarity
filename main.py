from gensim.models import KeyedVectors
import argparse
import pprint
import numpy as np
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from nltk.corpus import wordnet
from sklearn import svm
from sklearn import linear_model

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


def w2v_semantic_sim(input, vecs, output):
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


def getfeats(word, vecs, o, sent_num):
    """ This takes the word in question and
    the offset with respect to the instance
    word """
    o = str(o)
    vec_feat = ""
    if word in vecs:
        vec_feat = vecs[word]
    features = [
        # (o + 'word' + str(sent_num), word),
        # (o + 'capital' + str(sent_num), word[0].isupper()),
        # (o + 'word' + str(sent_num), str(vec_feat))
        # (o + 'accent', "é" in word or "ó" in word or "á" in word or "ú" or "í" in word
        # or "É" in word or "Ó" in word or "Á" in word or "Ú" or "Í" in word)
        # (o + 'alphanum', re.match('^[\w-]+$', word) is not None),
        # (o + 'cons', word[len(word) - 1] in 'qwrtypsdfghjklzxcvbnm')
        # (o + 'syllables', count_syllables(word))
    ]
    return features


def word2features(sent, vecs, i, sent_num):
    """ The function generates all features
    for the word at position i in the
    sentence."""
    features = []
    # the window around the token
    for o in [-1, 0, 1]:
        if i + o >= 0 and i + o < len(sent):
            word = sent[i + o][0]
            if word in vecs:
                featlist = getfeats(word, vecs, o, sent_num)
                features.extend(featlist)
    return dict(features)


def supervised_semantic_sim(train_input, val, test, vecs, output):

    train_feats = []
    train_sims = []

    for line in train_input:
        line_components = line.split("\t")
        sent1 = line_components[0]
        sent2 = line_components[1]
        words_in_sent1 = sent1.split()
        words_in_sent2 = sent2.split()

        for i in range(len(words_in_sent1)):
            feats = word2features(sent1,vecs,i,1)
            feat1 = feats

        for i in range(len(words_in_sent2)):
            feats = word2features(sent2,vecs,i,2)
            combined_feats = {**feat1, **feats}
            train_feats.append(combined_feats)
            train_sims.append(line_components[2][:-1])

    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(train_feats)

    # obtain model on training data

    model = LogisticRegression()
    model.fit(X_train, train_sims)

    # model = LogisticRegression(verbose=1, max_iter=10)
    # model.fit(X_train, train_sims)

    test_feats = []
    test_sims = []

    # predict on test data

    for line in test:
        line_components = line.split("\t")
        sent1 = line_components[0]
        sent2 = line_components[1]
        words_in_sent1 = sent1.split()
        words_in_sent2 = sent2.split()

        for i in range(len(words_in_sent1)):
            feats = word2features(sent1, vecs, i, 1)
            feat1 = feats

        for i in range(len(words_in_sent2)):
            feats = word2features(sent2, vecs, i, 2)
            combined_feats = {**feat1, **feats}
            test_feats.append(combined_feats)
            # train_sims.append(line_components[2][:-1])

    X_test = vectorizer.transform(test_feats)
    y_pred = model.predict(X_test)
    i = 0
    for line in test:
        line_components = line.split("\t")
        sent1 = line_components[0]
        sent2 = line_components[1]
        output.write(sent1+"\t"+sent2+"\t"+y_pred[i]+"\n")
        i += 1

    pass

def supervised_synonym_sim(train_input, val, test, vecs):

    train_feats = []
    train_sims = []
    for line in train_input:
        line_components = line.split("\t")
        sent1 = line_components[0]
        sent2 = line_components[1]
        words_in_sent1 = sent1.split()
        words_in_sent2 = sent2.split()

        synonym_cnt = 0
        antonym_cnt = 0

        for word in words_in_sent1:
            synonyms = []
            antonyms = []

            for syn in wordnet.synsets(word):
                for l in syn.lemmas():
                    synonyms.append(l.name())
                    if l.antonyms():
                        antonyms.append(l.antonyms()[0].name())

            for word2 in words_in_sent2:
                if word2 in synonyms:
                    synonym_cnt += 1
                elif word2 in antonyms:
                    antonym_cnt += 1

        v1 = np.zeros(vecs["hi"].shape)
        for word in words_in_sent1:
            if word in vecs:
                v1 = v1 + np.asarray(vecs[word])

        v2 = np.zeros(vecs["hi"].shape)
        for word in words_in_sent2:
            if word in vecs:
                v2 = v2 + np.asarray(vecs[word])

        sim = compute_cosine_similarity(v1, v2)

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
        v1_bag = np.zeros(n)
        v2_bag = np.zeros(n)
        for i in range(0, n):
            v1_bag[i] = tf_idf1[tf_idf[i]]
            v2_bag[i] = tf_idf2[tf_idf[i]]

        sim_bag = compute_cosine_similarity(v1_bag, v2_bag)

        train_feats.append(dict([('synonyms', synonym_cnt), ('antonyms', antonym_cnt), ('cos', sim), ('bag_cos', sim_bag)]))
        train_sims.append(float(line_components[2][:-1]))


    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(train_feats)

    # obtain model on training data

    model = svm.SVR()
    model.fit(X_train, train_sims)


    # predict on validation data

    val_feats = []


    for line in val:
        line_components = line.split("\t")
        sent1 = line_components[0]
        sent2 = line_components[1]
        words_in_sent1 = sent1.split()
        words_in_sent2 = sent2.split()

        synonym_cnt = 0
        antonym_cnt = 0

        for word in words_in_sent1:
            synonyms = []
            antonyms = []

            for syn in wordnet.synsets(word):
                for l in syn.lemmas():
                    synonyms.append(l.name())
                    if l.antonyms():
                        antonyms.append(l.antonyms()[0].name())

            for word2 in words_in_sent2:
                if word2 in synonyms:
                    synonym_cnt += 1
                elif word2 in antonyms:
                    antonym_cnt += 1

        v1 = np.zeros(vecs["hi"].shape)
        for word in words_in_sent1:
            if word in vecs:
                v1 = v1 + np.asarray(vecs[word])

        v2 = np.zeros(vecs["hi"].shape)
        for word in words_in_sent2:
            if word in vecs:
                v2 = v2 + np.asarray(vecs[word])

        sim = compute_cosine_similarity(v1, v2)

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
        v1_bag = np.zeros(n)
        v2_bag = np.zeros(n)
        for i in range(0, n):
            v1_bag[i] = tf_idf1[tf_idf[i]]
            v2_bag[i] = tf_idf2[tf_idf[i]]

        sim_bag = compute_cosine_similarity(v1_bag, v2_bag)

        val_feats.append(dict([('synonyms', synonym_cnt), ('antonyms', antonym_cnt), ('cos', sim), ('bag_cos', sim_bag)]))

    X_val = vectorizer.transform(val_feats)
    y_pred_val = model.predict(X_val)
    output_val = open("pred_val_ex2.txt","w")
    i = 0
    for line in val:
        line_components = line.split("\t")
        sent1 = line_components[0]
        sent2 = line_components[1]
        output_val.write(sent1+"\t"+sent2+"\t"+str(y_pred_val[i])+"\n")
        i += 1
    output_val.close()


    # predict on test data

    test_feats = []

    for line in test:
        line_components = line.split("\t")
        sent1 = line_components[0]
        sent2 = line_components[1]
        words_in_sent1 = sent1.split()
        words_in_sent2 = sent2.split()

        synonym_cnt = 0
        antonym_cnt = 0

        for word in words_in_sent1:
            synonyms = []
            antonyms = []

            for syn in wordnet.synsets(word):
                for l in syn.lemmas():
                    synonyms.append(l.name())
                    if l.antonyms():
                        antonyms.append(l.antonyms()[0].name())

            for word2 in words_in_sent2:
                if word2 in synonyms:
                    synonym_cnt += 1
                elif word2 in antonyms:
                    antonym_cnt += 1

        v1 = np.zeros(vecs["hi"].shape)
        for word in words_in_sent1:
            if word in vecs:
                v1 = v1 + np.asarray(vecs[word])

        v2 = np.zeros(vecs["hi"].shape)
        for word in words_in_sent2:
            if word in vecs:
                v2 = v2 + np.asarray(vecs[word])

            # else:
            #     min_dist = 10000
            #     best_match = ''
            #     count = 0
            #     tick = 0
            #     for v in vecs.vocab:
            #         tick += 1
            #         dist = edit_distance(word, v)
            #         if dist < min_dist:
            #             min_dist = dist
            #             best_match = v
            #             count += 1
            #             if count > 2 or min_dist < 4:
            #                 break
            #         if tick > 10000:
            #             break
            #     v1 = v1 + np.asarray(vecs[best_match])

        sim = compute_cosine_similarity(v1, v2)
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
        v1_bag = np.zeros(n)
        v2_bag = np.zeros(n)
        for i in range(0, n):
            v1_bag[i] = tf_idf1[tf_idf[i]]
            v2_bag[i] = tf_idf2[tf_idf[i]]

        sim_bag = compute_cosine_similarity(v1_bag, v2_bag)

        test_feats.append(dict([('synonyms', synonym_cnt), ('antonyms', antonym_cnt), ('cos', sim), ('bag_cos', sim_bag)]))

    X_test = vectorizer.transform(test_feats)
    y_pred = model.predict(X_test)
    output = open("pred_test_ex2.txt","w")
    i = 0
    for line in test:
        line_components = line.split("\t")
        sent1 = line_components[0]
        sent2 = line_components[1]
        output.write(sent1+"\t"+sent2+"\t"+str(y_pred[i])+"\n")
        i += 1
    output.close()
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

    with open("data/en-train.txt", 'r') as train_file:
        train_input = train_file.readlines()

    with open("data/en-val.txt", 'r') as val_file:
        val = val_file.readlines()

    with open("data/en-test.txt", 'r') as test_file:
        test = test_file.readlines()

    # Baseline
    # generates prediction for input textfile, 'pred_simple.txt'
    simple_baseline(input, output_simple)

    # Extension 1
    # generates prediction for input textfile, 'pred_ex1.txt'
    w2v_semantic_sim(input, vecs, output_w2v)

    # Extension 2 + More [tried several different models/features to achieve best results --> see writeup]
    # automatically generates predictions for validation and test set, 'pred_val_ex2.txt' and 'pred_test_ex2.txt'
    supervised_synonym_sim(train_input, val, test, vecs)
    

    # Close files
    output_simple.close()
    output_w2v.close()
    inputfile.close()
    train_file.close()
    val_file.close()
    test_file.close()