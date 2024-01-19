import numpy as np
import nltk
import wrench.dataset
from sklearn.metrics import accuracy_score
from LLMDP.data_utils import preprocess_text
from abc import ABC
import re


class AbstractLF(ABC):
    def __eq__(self, other):
        # check whether two LFs are equal
        pass

    def apply(self, x, **kwargs):
        # apply LF to an instance
        pass

    def get_active_indices(self, dataset):
        # get the indices of instances where the LF is activated
        pass

    def apply_to_dataset(self, dataset):
        # apply LF to a dataset and get weak labels
        pass

    def into(self):
        # return a string contain the information of this LF
        pass

    def get_cov_acc(self, dataset):
        preds = self.apply_to_dataset(dataset)
        active_indices = np.nonzero(preds != -1)[0]
        if len(active_indices) == 0:
            return 0.0, np.nan

        ys = np.array(dataset.labels)
        acc = accuracy_score(ys[active_indices], preds[active_indices])
        cov = len(active_indices) / len(dataset)
        return cov, acc


class KeywordLF(AbstractLF):
    def __init__(self, keyword, label, stop_words=None, stemming="porter"):
        self.keyword = keyword
        self.label = label
        self.stop_words = stop_words
        self.stemming = stemming

    def __eq__(self, other):
        if isinstance(other, KeywordLF):
            if self.keyword == other.keyword and self.label == other.label:
                return True

        return False

    def apply(self, x, **kwargs):
        processed_text = preprocess_text(x, stop_words=self.stop_words, stemming=self.stemming)
        if self.keyword in processed_text:
            return self.label
        else:
            return -1

    def apply_to_dataset(self, dataset):
        wl = np.repeat(-1, len(dataset))
        active_indices = self.get_active_indices(dataset)
        if len(active_indices) > 0:
            wl[active_indices] = self.label
        return wl

    def get_active_indices(self, dataset):
        # use pre-computed revert index to speed up
        if self.keyword in dataset.revert_index:
            active_indices = dataset.revert_index[self.keyword]
        else:
            active_indices = np.array([])

        return active_indices

    def info(self):
        return f"{self.keyword}->{self.label}"

    def isempty(self):
        return self.keyword.lower() == "none"


class RegexLF(AbstractLF):
    def __init__(self, regex, label):
        self.regex = regex
        self.label = label

    def __eq__(self, other):
        if isinstance(other, RegexLF):
            if self.regex == other.regex and self.label == other.label:
                return True

        return False

    def apply(self, x, **kwargs):
        regex = self.regex
        if "{{A}}" in regex and "entity1" in kwargs:
            regex = re.sub("\{\{A\}\}", re.escape(kwargs["entity1"]), regex)
        if "{{B}}" in regex and "entity2" in kwargs:
            regex = re.sub("\{\{B\}\}", re.escape(kwargs["entity2"]), regex)

        try:
            if re.search(regex, x, flags=re.I):
                return self.label
            else:
                return -1
        except re.error:
            return -1

    def apply_to_dataset(self, dataset):
        if isinstance(dataset, wrench.dataset.RelationDataset):
            wl = []
            for i in range(len(dataset)):
                text = dataset.examples[i]["text"]
                entity1 = dataset.examples[i]["entity1"]
                entity2 = dataset.examples[i]["entity2"]
                wl.append(self.apply(text, entity1=entity1, entity2=entity2))

        else:
            wl = [self.apply(dataset.examples[i]["text"]) for i in range(len(dataset))]

        return np.array(wl)

    def get_active_indices(self, dataset):
        wl = self.apply_to_dataset(dataset)
        active_indices = np.nonzero(wl != -1)[0]
        return active_indices

    def info(self):
        return f"{self.regex}->{self.label}"

    def isempty(self):
        return self.regex.lower() == "none"


def create_label_matrix(dataset, lfs):
    weak_labels = []
    for lf in lfs:
        wl = lf.apply_to_dataset(dataset)
        weak_labels.append(wl.reshape(-1,1))

    weak_labels = np.hstack(weak_labels)
    return weak_labels


def check_all_class(ys, cardinality):
    mask = np.repeat(False, cardinality)
    for label in ys:
        mask[label] = True

    return np.all(mask)
