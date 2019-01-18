import tokenization
import modeling
import json
import tensorflow as tf
import numpy as np
import pickle
from joblib import Parallel, delayed

stories = [story.rstrip('\n') for story in open("train.wp_target")]

tokenizer = tokenization.FullTokenizer(
    vocab_file='/home/one/PycharmProjects/seq2seq-with-attention/BERT/uncased_L-12_H-768_A-12/vocab.txt')
tokenizer2 = tokenization.BasicTokenizer()


# 272600
def prallel_me(i):
    # stories[i] = stories[i].rstrip('\n')
    stories[i] = stories[i].split("<newline> <newline>")
    for j in range(len(stories[i])):
        stories[i][j] = tokenizer.tokenize(stories[i][j])
        stories[i][j] = tokenizer.convert_tokens_to_ids(stories[i][j])


stories = Parallel(backend='multiprocessing', n_jobs=8)(delayed(prallel_me)(i) for i in range(len(stories)))

print(stories[0][2])
print(stories[45][4])
print(stories[272599][7])

with open('stories.pkl', 'wb') as f:
    pickle.dump(stories, f)
    f.close()

# with open("stories.pkl",'rb') as file:
#     object_file = pickle.load(file)
#     file.close()
#
# print(object_file[i][j])