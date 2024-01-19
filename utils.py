import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score
from snorkel.labeling.analysis import LFAnalysis
import LLMDP.logconfig
import logging 

logger = logging.getLogger(__name__)

def print_dataset_stats(dataset, split="train"):
    print("{} size: {}".format(split, len(dataset)))
    values, counts = np.unique(dataset.labels, return_counts=True)
    freq = counts / len(dataset)
    print("Label distribution: ", freq)


def evaluate_lfs(labels, L_train, lf_classes=None, n_class=2):
    # evaluate the coverage, accuracy of label functions
    n_lf = L_train.shape[1]
    n_active = np.sum(L_train != -1, axis=0)
    n_correct = np.sum(L_train == labels.reshape(-1,1), axis=0)
    lf_covs = n_active / len(labels)
    lf_accs = np.divide(n_correct, n_active, out=np.repeat(np.nan, len(n_active)), where=n_active!=0)
    lf_cov_avg = np.mean(lf_covs)
    lf_acc_avg = np.mean(lf_accs[lf_covs != 0])
    # evaluate conflicts, overlaps
    lf_stats = LFAnalysis(L_train).lf_summary(Y=labels)
    lf_acc_avg_2 = lf_stats["Emp. Acc."].mean()
    lf_cov_avg_2 = lf_stats["Coverage"].mean()
    lf_overlap_avg = lf_stats["Overlaps"].mean()
    lf_conflict_avg = lf_stats["Conflicts"].mean()
    results = {
        "n_lf": n_lf,
        "lf_cov_avg": lf_cov_avg,
        "lf_acc_avg": lf_acc_avg,
        "lf_overlap_avg": lf_overlap_avg,
        "lf_conflict_avg": lf_conflict_avg,
    }

    # evaluate the LF quality per class
    if lf_classes is not None:
        lf_num_pc = []
        lf_acc_avg_pc = []
        lf_cov_avg_pc = []
        lf_cov_total_pc = []
        for c in range(n_class):
            active_lfs = lf_classes == c
            logger.warning(f"number of active_lfs for class {c}: {active_lfs}")
            lf_num_pc.append(np.sum(active_lfs))
            if np.sum(active_lfs) != 0:
                logger.warning(f"active_lfs lf_accs: {lf_accs[active_lfs]}")
                lf_acc_avg_pc.append(np.nanmean(lf_accs[active_lfs]))
                logger.warning(f"active_lfs lf_covs: {lf_accs[active_lfs]}")
                lf_cov_avg_pc.append(np.mean(lf_covs[active_lfs]))
                L_train_pc = L_train[:, active_lfs]
                cov_pc = np.mean(np.max(L_train_pc, axis=1) == c)
                lf_cov_total_pc.append(cov_pc)
            else:
                # no LF emits label for class c
                lf_acc_avg_pc.append(np.nan)
                lf_cov_avg_pc.append(np.nan)
                lf_cov_total_pc.append(np.nan)

        results["n_lf_per_class"] = lf_num_pc
        results["lf_acc_per_class"] = lf_acc_avg_pc
        results["lf_cov_per_class"] = lf_cov_avg_pc
        results["lf_cov_total_per_class"] = lf_cov_total_pc


    return results


def evaluate_labels(labels, preds, n_class=2):
    # Evaluate the prediction results. -1 in preds mean the label model rejects making prediction.
    covered_indices = preds != -1
    covered_labels = labels[covered_indices]
    covered_preds = preds[covered_indices]
    if -1 in covered_labels:
        results = {
            "coverage": np.sum(covered_indices) / len(preds),
            "accuracy": np.nan,
        }
        return results
    # label quality summary on covered part
    if n_class == 2:
        average = "binary"
    else:
        average = "macro"
    coverage = np.sum(covered_indices) / len(preds)
    accuracy = accuracy_score(covered_labels, covered_preds)
    precision = precision_score(covered_labels, covered_preds, average=average)
    recall = recall_score(covered_labels, covered_preds, average=average)
    f1 = f1_score(covered_labels, covered_preds, average=average)

    # label distribution and predicted label distribution
    label_dist = np.zeros(n_class, dtype=float)
    pred_label_dist = np.zeros(n_class, dtype=float)
    for c in range(n_class):
        label_dist[c] = np.sum(labels == c) / len(labels)
        pred_label_dist[c] = np.sum(covered_preds == c) / len(covered_preds)
    # class-specific label quality
    covered_precision_perclass = precision_score(covered_labels, covered_preds, average=None)
    covered_recall_perclass = recall_score(covered_labels, covered_preds, average=None)
    covered_f1_perclass = f1_score(covered_labels, covered_preds, average=None)
    # confusion matrix
    cm = confusion_matrix(covered_labels, covered_preds)
    results = {
        "coverage": coverage,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "label_distribution": label_dist.tolist(),
        "pred_distribution": pred_label_dist.tolist(),
        "per_class_precision": covered_precision_perclass.tolist(),
        "per_class_recall": covered_recall_perclass.tolist(),
        "per_class_f1": covered_f1_perclass.tolist(),
        "confusion_matrix": cm
    }
    return results


def evaluate_disc_model(disc_model, test_dataset):
    y_pred = disc_model.predict(test_dataset.features)
    y_probs = disc_model.predict_proba(test_dataset.features)
    test_acc = accuracy_score(test_dataset.labels, y_pred)
    logger.warning("accuracy per class using downstream model : raw data")

    if test_dataset.n_class == 2:
        test_auc = roc_auc_score(test_dataset.labels, y_probs[:, 1])
        test_f1 = f1_score(test_dataset.labels, y_pred)
    else:
        logger.warning("test_dataset.labels")
        logger.warning(test_dataset.labels)
        logger.warning(f"len(test_dataset.labels : {len(test_dataset)}")
        logger.warning(f"unique labels : {set(test_dataset.labels)}")
        logger.warning(f"len(unique_labels) : {len(set(test_dataset.labels))}")
        logger.warning("y_probs")
        logger.warning(y_probs)
        logger.warning(f"len y_probs : {len(y_probs)}")
        logger.warning(f"len y_probs[0]: {len(y_probs[0])}")
        test_auc = roc_auc_score(test_dataset.labels, y_probs, average="macro", multi_class="ovo")
        test_f1 = f1_score(test_dataset.labels, y_pred, average="macro")

    cm = confusion_matrix(test_dataset.labels, y_pred)
    print("Confusion matrix (end model):\n", cm)

    results = {
        "acc": test_acc,
        "auc": test_auc,
        "f1": test_f1
    }
    return results
