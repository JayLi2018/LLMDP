import abc
import numpy as np
import random
from alipy import ToolBox
from sentence_transformers import SentenceTransformer, util
from scipy.stats import entropy
from LLMDP.data_utils import build_revert_index
from pathlib import Path
import LLMDP.logconfig
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def get_sampler(train_dataset, sampler_type, **kwargs):
    if sampler_type in ["passive", "uncertain", "QBC"]:
        return ActiveSampler(train_dataset, al_method=sampler_type, **kwargs)
    elif sampler_type == "SEU":
        return SEUSampler(train_dataset, **kwargs)
    elif sampler_type == "weighted":
        return WeightedScoreSampler(train_dataset, **kwargs)
    elif sampler_type == "uniform":
        return UniformSampler(train_dataset, **kwargs)
    else:
        raise ValueError("Sampler not supported")


class Sampler(abc.ABC):
    def __init__(self, dataset, **kwargs):
        self.dataset = dataset

    def sample(self, batch_size=1, **kwargs):
        pass

class UniformSampler(Sampler):
    # a simple sampler that ensures for every class that exists in the 
    # example, there will be the same number of samples being selected

    def __init__(self, dataset, sample_instance_per_class, seed=0, **kwargs):
        # label_class_name, sample_instance_per_class, 
        self.dataset = dataset
        logger.warning(f"len(dataset.labels): {len(dataset.labels)}")
        self.available_classes_to_sample_from = {l: sample_instance_per_class for l in list(set(dataset.labels))}
        self.label_and_indices = defaultdict(list)
        for i, j in enumerate(self.dataset.labels):
            self.label_and_indices[j].append(i)
        logger.warning(f"uniform sampler available classes: {self.available_classes_to_sample_from}")
        logger.warning(f"label and indices: {self.label_and_indices}")
        
    
    def sample(self, batch_size=1, label_model=None, end_model=None):
        available_to_sample = [k for k,v in self.available_classes_to_sample_from.items() if v!=0]
        selected_key = random.choice(available_to_sample)
        # Randomly select a value from the corresponding list
        indices = random.sample(self.label_and_indices[selected_key], 2)
        self.label_and_indices[selected_key] = [v for v in self.label_and_indices[selected_key] if v not in indices]
        self.available_classes_to_sample_from[selected_key]-=batch_size
        return indices


class ActiveSampler(Sampler):
    """
    Sampler that leverages off-the-box active learning methods
    """
    def __init__(self, dataset, al_method="uncertain", seed=0, **kwargs):
        super(ActiveSampler, self).__init__(dataset)
        self.al_method = al_method
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.alibox = ToolBox(X=dataset.features, y=dataset.labels, query_type="AllLabels")
        self.label_index = self.alibox.IndexCollection([0]).difference_update([0])
        # logger.warning(f"label_index: {self.label_index}")
        self.unlabel_index = self.alibox.IndexCollection(np.arange(len(dataset)))

    def sample(self, batch_size=1, label_model=None, end_model=None):
        if end_model is None or self.al_method == "passive":
            strategy = self.alibox.get_query_strategy(strategy_name="QueryInstanceRandom")
        elif self.al_method == "uncertain":
            strategy = self.alibox.get_query_strategy(strategy_name="QueryInstanceUncertainty")
        elif self.al_method == "QBC":
            strategy = self.alibox.get_query_strategy(strategy_name="QueryInstanceQBC")
        else:
            raise ValueError("AL method not supported.")

        indices = strategy.select(label_index=self.label_index,
                                  unlabel_index=self.unlabel_index,
                                  model=end_model,
                                  batch_size=batch_size)

        self.label_index.update(indices)
        self.unlabel_index.difference_update(indices)
        return indices


class SEUSampler(Sampler):
    """
    Select by expected utility (Nemo)
    """
    def __init__(self, dataset, **kwargs):
        super(SEUSampler, self).__init__(dataset)
        # if hasattr(dataset, "revert_index"):
        #     self.revert_index = dataset.revert_index
        # else:
        #     self.revert_index = build_revert_index(dataset)
        self.revert_index = build_revert_index(dataset, cache_path=kwargs["index_path"])

        self.active_keyword_index = [[] for _ in range(len(dataset))]
        for (keyword_index, keyword) in enumerate(self.revert_index):
            active_index = self.revert_index[keyword]
            for i in active_index:
                self.active_keyword_index[i].append(keyword_index)

        self.n_class = dataset.n_class
        self.label_index = np.array([], dtype=int)
        self.unlabel_index = np.arange(len(dataset), dtype=int)

    def sample(self, batch_size=1, label_model=None, end_model=None):
        if label_model is None:
            sampled_index = np.random.choice(self.unlabel_index, size=batch_size, replace=False)
        else:
            lm_probs = label_model.predict_proba(np.array(self.dataset.weak_labels))
            y_preds = label_model.predict(np.array(self.dataset.weak_labels), tie_break_policy="random")
            lf_accs = np.zeros((len(self.revert_index), self.n_class), dtype=float)
            # estimate LF accuracy
            for (keyword_index, keyword) in enumerate(self.revert_index):
                preds = y_preds[self.revert_index[keyword]]
                pred_dist = np.bincount(preds, minlength=self.n_class) / len(preds)
                lf_accs[keyword_index, :] = pred_dist

                # for c in range(self.n_class):
                #     lf_acc = np.mean(y_preds[self.revert_index[keyword]] == c)
                #     lf_accs[keyword_index, c] = lf_acc

            # estimate LF utility
            lf_scores = np.zeros_like(lf_accs, dtype=float)
            uncertain_score = entropy(lm_probs, axis=1)
            for (keyword_index, keyword) in enumerate(self.revert_index):
                active_indices = self.revert_index[keyword]
                base_score = - np.sum(uncertain_score[active_indices])
                for c in range(self.n_class):
                    pos_indices = active_indices[y_preds[active_indices] == c]
                    lf_scores[keyword_index, c] = 2 * np.sum(uncertain_score[pos_indices]) - base_score

            # compute score per instance. Select a random subset for evaluation to speed up
            if len(self.unlabel_index) > 3000:
                candidate_index = np.random.choice(self.unlabel_index, size=3000, replace=False)
            else:
                candidate_index = self.unlabel_index

            unlabel_scores = []
            for idx in candidate_index:
                lf_acc_list = []
                lf_score_list = []
                for keyword_index in self.active_keyword_index[idx]:
                    for c in range(self.n_class):
                        lf_acc_list.append(lf_accs[keyword_index, c])
                        lf_score_list.append(lf_scores[keyword_index, c])

                lf_probs = np.array(lf_acc_list) / np.sum(lf_acc_list)
                score = np.sum(np.array(lf_score_list) * lf_probs)
                unlabel_scores.append(score)

            unlabel_pos = np.argsort(unlabel_scores)[-1:-batch_size-1:-1]
            sampled_index = candidate_index[unlabel_pos]

        self.label_index = np.union1d(self.label_index, sampled_index)
        self.unlabel_index = np.setdiff1d(self.unlabel_index, sampled_index)
        return sampled_index


class WeightedScoreSampler(Sampler):
    def __init__(self, dataset, embedding_model="all-MiniLM-L12-v2", distance="cosine",
                 uncertain_metric="entropy", k=100, alpha=1.0, beta=1.0, gamma=1.0, **kwargs):
        super(WeightedScoreSampler, self).__init__(dataset)
        self.embedding_model = SentenceTransformer(embedding_model)
        sentences = [dataset.examples[i]["text"] for i in range(len(dataset))]
        self.embeddings = self.embedding_model.encode(sentences)
        if distance == "cosine":
            self.sim_func = util.cos_sim
        else:
            raise ValueError("Similarity function not supported.")

        self.dist_mat = 1 - self.sim_func(self.embeddings, self.embeddings).numpy()
        self.k = k  # neighbor count
        self.alpha = alpha  # weight factor for uncertain score
        self.beta = beta  # weight factor for class balance
        self.gamma = gamma  # weight factor for distance to labeled set
        self.uncertain_metric = uncertain_metric
        # compute k-nearest neighbors for each point (including itself)
        self.neighbors = np.argsort(self.dist_mat, axis=1)[:,:k]
        # self.neighbor_dists = np.sort(self.dist_mat, axis=1)[:,:k]
        self.label_index = np.array([], dtype=int)
        self.unlabel_index = np.arange(len(dataset), dtype=int)
        self.kwargs = kwargs

    def distance_to_labeled(self, indices):
        if len(self.label_index) == 0:
            return np.ones(len(indices))
        distance = self.dist_mat[np.ix_(indices, self.label_index)]
        sorted_distance = np.sort(distance, axis=1)
        dist = sorted_distance[:, 0]
        dist[dist < 0.0] = 0.0  # avoid computation issue
        return dist

    def update(self, weak_labels):
        self.dataset.weak_labels = weak_labels

    def sample(self, batch_size=1, label_model=None, end_model=None):
        if label_model is None and end_model is None:
            probs = np.ones((len(self.dataset), self.dataset.n_class)) / self.dataset.n_class
        elif end_model is None:
            probs = label_model.predict_proba(np.array(self.dataset.weak_labels))
        else:
            probs = end_model.predict_proba(self.dataset.features)

        if self.uncertain_metric == "entropy":
            uncertain_score = entropy(probs, axis=1)
        elif self.uncertain_metric == "confidence":
            uncertain_score = 1 - np.max(probs, axis=1)
        elif self.uncertain_metric == "margin":
            sorted_probs = np.sort(probs, axis=1)
            uncertain_score = 1 - (sorted_probs[:,-1] - sorted_probs[:,-2])
        else:
            raise ValueError("Uncertainty score not supported.")

        if "class_balance" in self.kwargs and label_model is not None:
            desired_class_balance = self.kwargs["class_balance"]
            cur_preds = label_model.predict(np.array(self.dataset.weak_labels))
            pred_count = np.bincount(cur_preds, minlength=self.dataset.n_class)
            pred_class_balance = pred_count / len(self.dataset)
            class_ratio = desired_class_balance / (pred_class_balance+1e-6)
            class_ratio = class_ratio / np.sum(class_ratio)
            if end_model is not None:
                probs = end_model.predict_proba(self.dataset.features)
            else:
                probs = label_model.predict_proba(np.array(self.dataset.weak_labels))

            class_score = np.sum(probs * class_ratio, axis=1)
        else:
            class_score = np.ones(len(self.dataset))

        distance_score = self.distance_to_labeled(np.arange(len(self.dataset)))
        weights = uncertain_score * self.alpha + class_score * self.beta + distance_score * self.gamma
        neighbor_weights = weights[self.neighbors]
        scores = np.sum(neighbor_weights, axis=1)
        unlabel_scores = scores[self.unlabel_index]
        unlabel_pos = np.argsort(unlabel_scores)[-1:-batch_size-1:-1]
        sampled_index = self.unlabel_index[unlabel_pos]
        self.label_index = np.union1d(self.label_index, sampled_index)
        self.unlabel_index = np.setdiff1d(self.unlabel_index, sampled_index)
        return sampled_index





