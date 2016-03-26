from __future__ import division
from nltk.stem.porter import *

import pandas as pd
import numpy as np
import pickle
import sys
import inspect

import utils
import spellingCorrector

np.random.seed(123)
train_data_path ="../data/train.csv"
attributes_path = "../data/attributes.csv"
product_description_path = "../data/product_descriptions.csv"
test_data_path = "../data/test.csv"


train_data = pd.read_csv(train_data_path, encoding="ISO-8859-1")
train_data = train_data[(train_data.relevance != 2.50) & (train_data.relevance != 2.25) & (train_data.relevance != 2.75) & (train_data.relevance != 1.75) & (train_data.relevance != 1.50) & (train_data.relevance != 1.25)]
test_data = pd.read_csv(test_data_path, encoding="ISO-8859-1")
test_data['relevance'] = -1 * np.ones(test_data.shape[0])

productionDescription = pd.read_csv(product_description_path, encoding="ISO-8859-1")

df_all = pd.concat([train_data, test_data], axis=0, ignore_index=True)
df_all = pd.merge(df_all, productionDescription, how='left', on='product_uid')



def basic_features():

    # Spelling corrector
    df_all['search_term'] = df_all['search_term'].apply(lambda x : spellingCorrector.correctSpelling(x))
    # Remove stop words
    df_all['product_description'] = df_all['product_description'].apply(lambda x: utils.remove_stop_words(x))
    df_all['product_title'] = df_all['product_title'].apply(lambda x: utils.remove_stop_words(x))
    df_all['search_term'] = df_all['search_term'].apply(lambda x: utils.remove_stop_words(x))

    # Stem words
    df_all['product_description'] = df_all['product_description'].apply(lambda x: utils.str_stem(x))
    # df_all['product_title'] = df_all['product_title'].apply(lambda x: utils.str_stem(x))
    # df_all['search_term'] = df_all['search_term'].apply(lambda x: utils.str_stem(x))

    # print "computing " + inspect.currentframe().f_code.co_name
    # df_all['title_similarity'] = df_all.apply(lambda x : utils.jaccard_similarity(x['product_title'], x['search_term']), axis=1)
    # df_all['product_desc_similarity'] = df_all.apply(lambda x : utils.jaccard_similarity(x['product_description'], x['search_term']), axis=1)
    # df_all['title_similarity_common'] = df_all.apply(lambda x : utils.common_words(x['product_title'], x['search_term']), axis=1)
    # df_all['product_desc_similarity_common'] = df_all.apply(lambda x : utils.common_words(x['product_description'], x['search_term']), axis=1)
    # df_all['description_length'] = df_all.apply(lambda x :len(x.product_description), axis=1)
    # df_all['search_length'] = df_all.apply(lambda x :len(x.search_term), axis=1)
    # fl = open("../output/features/" + inspect.currentframe().f_code.co_name + ".pkl", 'w')
    # pickle.dump(df_all, fl)

def main():
    basic_features()


if __name__ == '__main__':
    main()
