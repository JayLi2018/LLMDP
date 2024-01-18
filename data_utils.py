import re
import numpy as np
import html
import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, SnowballStemmer
import xml.etree.ElementTree as ET
import wrench.dataset
import argparse
from wrench.dataset import load_dataset, BaseDataset
import pdb
from pathlib import Path
import os
import json


def preprocess_text(text, stop_words=None, stemming="porter"):
    if stop_words is not None:
        stop_words = set(stopwords.words(stop_words))
    else:
        stop_words = set()

    if stemming == "porter":
        stemmer = PorterStemmer()
    elif stemming == "snowball":
        stemmer = SnowballStemmer(language="english")
    else:
        stemmer = None

    processed_tokens = []
    tokens = nltk.word_tokenize(text.lower())
    for token in tokens:
        # filter out stopwords and non-words
        if token in stop_words or (re.search("^\w+$", token) is None):
            continue
        if stemmer is not None:
            token = stemmer.stem(token)

        processed_tokens.append(token)

    processed_text = " ".join(processed_tokens)
    return processed_text


def build_revert_index(dataset, stop_words=None, stemming="porter", max_ngram=1, cache_path=None):
    """
    Build reverted index for the given text dataset
    """
    if cache_path is not None:
        if os.path.exists(cache_path):
            with open(cache_path) as infile:
                reverted_index = json.load(infile)
                for phrase in reverted_index:
                    reverted_index[phrase] = np.array(reverted_index[phrase])

                return reverted_index

    # preprocess data
    corpus = [dataset.examples[idx]["text"] for idx in range(len(dataset))]
    if stop_words is not None:
        stop_words = set(stopwords.words(stop_words))
    else:
        stop_words = set()

    if stemming == "porter":
        stemmer = PorterStemmer()
    elif stemming == "snowball":
        stemmer = SnowballStemmer(language="english")
    else:
        stemmer = None

    reverted_index = {}

    for idx, text in enumerate(corpus):
        processed_tokens = []
        text = re.sub("\ufeff", "", text)
        tokens = nltk.word_tokenize(text.lower())
        for token in tokens:
            # filter out stopwords and non-words
            if token in stop_words or (re.search("^\w+$", token) is None):
                continue
            if stemmer is not None:
                token = stemmer.stem(token)

            processed_tokens.append(token)

        for n in range(max_ngram):
            phrases = ngrams(processed_tokens, n+1)
            for t in phrases:
                phrase = " ".join(t)
                if phrase in reverted_index:
                    if reverted_index[phrase][-1] != idx:
                        reverted_index[phrase].append(idx)
                else:
                    reverted_index[phrase] = [idx]

    if cache_path is not None:
        with open(cache_path, "w") as outfile:
            json.dump(reverted_index, outfile)

    for phrase in reverted_index:
        reverted_index[phrase] = np.array(reverted_index[phrase])

    return reverted_index


def load_wrench_data(data_root, dataset_name, feature, stopwords=None, stemming="porter", max_ngram=1, revert_index=True, append_cdr=False):
    if dataset_name == "cdr" and append_cdr:
        # align snippets in cdr dataset to original data
        train_dataset, valid_dataset, test_dataset = load_cdr_data(data_root, feature)
    elif feature in ["tfidf", "bow"]:
        train_dataset, valid_dataset, test_dataset = load_dataset(data_root, dataset_name,
                                                                  extract_feature=True, extract_fn=feature)
    else:
        train_dataset, valid_dataset, test_dataset = load_dataset(data_root, dataset_name,
                                                                  extract_feature=True, extract_fn=feature, cache_name=feature)


    if revert_index:
        train_cache_path = Path(data_root) / dataset_name / "train_index.json"
        valid_cache_path = Path(data_root) / dataset_name / "valid_index.json"
        test_cache_path = Path(data_root) / dataset_name / "test_index.json"
        train_dataset.revert_index = build_revert_index(train_dataset, stop_words=stopwords, stemming=stemming,
                                                        max_ngram=max_ngram, cache_path=train_cache_path)
        valid_dataset.revert_index = build_revert_index(valid_dataset, stop_words=stopwords, stemming=stemming,
                                                        max_ngram=max_ngram, cache_path=valid_cache_path)
        test_dataset.revert_index = build_revert_index(test_dataset, stop_words=stopwords, stemming=stemming,
                                                       max_ngram=max_ngram, cache_path=test_cache_path)

    return train_dataset, valid_dataset, test_dataset


def load_cdr_data(data_root, feature, corpus_path="cdr/CDR_corpus"):
    """
    Load cdr data and align with original corpus
    """
    dataset_name = "cdr"
    train_dataset, valid_dataset, test_dataset = load_dataset(data_root, dataset_name, extract_feature=False)
    corpus = []
    for filename in ["CDR_TrainingSet.BioC.xml", "CDR_DevelopmentSet.BioC.xml", "CDR_TestSet.BioC.xml"]:
        filepath = Path(data_root) / corpus_path / filename
        tree = ET.parse(filepath)
        root = tree.getroot()
        for doc in root.iter("document"):
            passages = doc.findall("passage")
            title = html.unescape(passages[0].find("text").text)
            abstract = html.unescape(passages[1].find("text").text)
            text = "{}\n{}".format(title, abstract)
            corpus.append(text)

    for dataset in [train_dataset, valid_dataset, test_dataset]:
        for idx in range(len(dataset)):
            snippet = dataset.examples[idx]["text"]
            res = [x for x in corpus if x.find(snippet) != -1]
            if len(res) == 1:
                dataset.examples[idx]["text_snippet"] = snippet
                dataset.examples[idx]["text"] = res[0]
                start_pos = res[0].find(snippet)
                st_1 = start_pos + dataset.examples[idx]["span1"][0]
                ed_1 = start_pos + dataset.examples[idx]["span1"][1]
                st_2 = start_pos + dataset.examples[idx]["span2"][0]
                ed_2 = start_pos + dataset.examples[idx]["span2"][1]
                assert res[0][st_1:ed_1] == dataset.examples[idx]["entity1"]
                assert res[0][st_2:ed_2] == dataset.examples[idx]["entity2"]
                dataset.examples[idx]["span1"] = [st_1, ed_1]
                dataset.examples[idx]["span2"] = [st_2, ed_2]
            elif len(res) > 1:
                print("More than one matches found.")
                print("Snippet: ", snippet)
                for i, text in enumerate(res):
                    print(f"Match {i+1}: ", text)
            else:
                print("No match found for snippet:", snippet)

    extractor_fn = train_dataset.extract_feature(extract_fn=feature, return_extractor=True, cache_name=feature)
    valid_dataset.extract_feature(extract_fn=extractor_fn, return_extractor=False, cache_name=feature)
    test_dataset.extract_feature(extract_fn=extractor_fn, return_extractor=False, cache_name=feature)
    return train_dataset, valid_dataset, test_dataset