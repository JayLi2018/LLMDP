import argparse
from data_utils import load_wrench_data
from sampler import get_sampler
from lf_agent import get_lf_agent
from lf_family import create_label_matrix
from label_model import get_wrench_label_model, is_valid_snorkel_input
from end_model import train_disc_model
from wrench.search import grid_search
from wrench.search_space import SEARCH_SPACE
from utils import print_dataset_stats, evaluate_lfs, evaluate_labels, evaluate_disc_model
from gpt_utils import create_user_prompt
import numpy as np
from sklearn.metrics import accuracy_score
import wandb
import pprint
from pathlib import Path
import pickle

def main(args):
    train_dataset, valid_dataset, test_dataset = load_wrench_data(data_root=args.dataset_path,
                                                                  dataset_name=args.dataset_name,
                                                                  feature=args.feature_extractor,
                                                                  stopwords=args.stop_words,
                                                                  stemming=args.stemming,
                                                                  max_ngram=args.max_ngram,
                                                                  revert_index=args.lf_type == "keyword",
                                                                  append_cdr=args.append_cdr)
    # 1st step : load data

    print(f"Dataset path: {args.dataset_path}, name: {args.dataset_name}")
    print_dataset_stats(train_dataset, split="train")
    print_dataset_stats(valid_dataset, split="valid")
    print_dataset_stats(test_dataset, split="test")

    if args.save_wandb:
        group_id = wandb.util.generate_id()
        config_dict = vars(args)
        config_dict["method"] = "LLMDP"
        config_dict["group_id"] = group_id

    if args.dataset_name == "medical_abstract_unique":
        args.dataset_name = "medical_abstract"

    rng = np.random.default_rng(args.seed)
    runs_and_lfs = {}
    for run in range(args.runs):
        if args.save_wandb:
            wandb.init(
                project="LLMDP",
                config=config_dict
            )

        if args.lf_agent == "wrench":
            seed = rng.choice(10000)
            # use the LFs provided by wrench
            L_train = np.array(train_dataset.weak_labels)
            train_labels = np.array(train_dataset.labels)
            lf_train_stats = evaluate_lfs(train_labels, L_train, n_class=train_dataset.n_class)

            label_model = get_wrench_label_model(args.label_model, verbose=False)
            search_space = SEARCH_SPACE.get(args.label_model, None)
            if search_space is not None and args.tune_label_model:
                if args.tune_metric == "f1":
                    metric = "f1_binary" if train_dataset.n_class == 2 else "f1_macro"
                else:
                    metric = "acc"
                searched_paras = grid_search(label_model, dataset_train=train_dataset, dataset_valid=valid_dataset,
                                             metric=metric, direction='auto',
                                             search_space=search_space,
                                             n_repeats=1, n_trials=100, parallel=False)
                label_model = get_wrench_label_model(args.label_model, **searched_paras, verbose=False)
            label_model.fit(dataset_train=train_dataset,
                            dataset_valid=valid_dataset,
                            )
            ys_tr = label_model.predict(train_dataset)
            ys_tr_soft = label_model.predict_proba(train_dataset)

            train_covered_indices = (np.max(L_train, axis=1) != -1) & (ys_tr != -1)  # indices covered by LFs
            ys_tr[~train_covered_indices] = -1
            train_label_stats = evaluate_labels(train_labels, ys_tr, n_class=train_dataset.n_class)

            # train end model
            xs_tr = train_dataset.features[train_covered_indices, :]
            ys_tr = ys_tr[train_covered_indices]
            ys_tr_soft = ys_tr_soft[train_covered_indices, :]
            disc_model = train_disc_model(model_type=args.end_model,
                                          xs_tr=xs_tr,
                                          ys_tr_soft=ys_tr_soft,
                                          ys_tr_hard=ys_tr,
                                          valid_dataset=valid_dataset,
                                          soft_training=args.use_soft_labels,
                                          tune_end_model=args.tune_end_model,
                                          tune_metric=args.tune_metric,
                                          seed=seed)
            test_perf = evaluate_disc_model(disc_model, test_dataset)

            if args.save_wandb:
                wandb.run.summary["lf_num"] = L_train.shape[1]
                wandb.run.summary["lf_acc_avg"] = lf_train_stats["lf_acc_avg"]
                wandb.run.summary["lf_cov_avg"] = lf_train_stats["lf_cov_avg"]
                wandb.run.summary["lf_overlap_avg"] = lf_train_stats["lf_overlap_avg"]
                wandb.run.summary["lf_conflict_avg"] = lf_train_stats["lf_conflict_avg"]
                wandb.run.summary["train_precision"] = train_label_stats["accuracy"]
                wandb.run.summary["train_coverage"] = train_label_stats["coverage"]
                wandb.run.summary["test_acc"] = test_perf["acc"]
                wandb.run.summary["test_f1"] = test_perf["f1"]
                wandb.run.summary["test_auc"] = test_perf["auc"]
                wandb.finish()

            continue

        seed = rng.choice(10000)
        values, counts = np.unique(valid_dataset.labels, return_counts=True)
        class_balance = counts / len(valid_dataset)
        sampler = get_sampler(train_dataset=train_dataset,
                              sampler_type=args.sampler,
                              embedding_model=args.embedding_model,
                              distance=args.distance,
                              uncertain_metric=args.uncertain_metric,
                              k=args.neighbor_num,
                              alpha=args.alpha,
                              beta=args.beta,
                              gamma=args.gamma,
                              class_balance=class_balance,
                              seed=seed,
                              index_path=Path(args.dataset_path) / args.dataset_name / "train_index_unigram.json"
                              )

        lf_agent = get_lf_agent(train_dataset=train_dataset,
                                valid_dataset=valid_dataset,
                                agent_type=args.lf_agent,
                                dataset_name=args.dataset_name,
                                # LLM prompt arguments
                                model=args.lf_llm_model,
                                example_per_class=args.example_per_class,
                                example_selection=args.example_selection,
                                return_explanation=args.return_explanation,
                                temperature=args.temperature,
                                top_p=args.top_p,
                                n_completion=args.n_completion,
                                # LF creation arguments
                                lf_type=args.lf_type,
                                filter_methods=args.lf_filter,
                                acc_threshold=args.lf_acc_threshold,
                                overlap_threshold=args.lf_overlap_threshold,
                                seed=seed,
                                stop_words=args.stop_words,
                                stemming=args.stemming,
                                max_ngram=args.max_ngram,
                                max_lf_per_iter=args.max_lf_per_iter,
                                sleep_time=args.sleep_time,
                                )
        label_model = None
        disc_model = None
        L_train = None
        L_val = None
        best_valid_perf = 0.0
        tolerance = 0
        searched_paras = {}
        lfs = []
        gt_labels = []
        response_labels = []
        start = 0  # the number of LFs applied in previous iteration

        for t in range(args.num_query):
            query_idx = sampler.sample(label_model=label_model, end_model=disc_model)[0]
            if args.display:
                query = create_user_prompt("", args.dataset_name, train_dataset, query_idx)
                print("\nQuery [{}]: {}".format(query_idx, query))

            cur_lfs = lf_agent.create_lf(query_idx)
            if args.display:
                for lf in cur_lfs:
                    print("LF: ", lf.info())
                print("Ground truth: ", train_dataset.labels[query_idx])

            gt_labels.append(train_dataset.labels[query_idx])
            if cur_lfs[0].label in range(train_dataset.n_class):
                response_labels.append(cur_lfs[0].label)
            else:
                response_labels.append(-1)

            for lf in cur_lfs:
                if not lf.isempty():
                    lfs.append(lf)

            if len(lfs) > start:
                # calculate weak label matrix
                if start == 0:
                    L_train = create_label_matrix(train_dataset, lfs)
                    L_val = create_label_matrix(valid_dataset, lfs)
                else:
                    L_train_new = create_label_matrix(train_dataset, lfs[start:])
                    L_val_new = create_label_matrix(valid_dataset, lfs[start:])
                    L_train = np.concatenate((L_train, L_train_new), axis=1)
                    L_val = np.concatenate((L_val, L_val_new), axis=1)
                    start = len(lfs)

                # update weak label matrix
                train_dataset.weak_labels = L_train.tolist()
                valid_dataset.weak_labels = L_val.tolist()
                if is_valid_snorkel_input(lfs):
                    label_model = get_wrench_label_model(args.label_model, verbose=False, **searched_paras)
                else:
                    label_model = get_wrench_label_model("MV")

                label_model.fit(dataset_train=train_dataset,
                                dataset_valid=valid_dataset,
                                )

            if t % args.train_iter == args.train_iter - 1:
                if len(lfs) == 0:
                    response_acc = accuracy_score(gt_labels, response_labels)
                    cur_result = {
                        "num_query": t + 1,
                        "lf_num": 0,
                        "lf_acc_avg": np.nan,
                        "lf_cov_avg": np.nan,
                        "response_acc": response_acc,
                        "train_precision": np.nan,
                        "train_coverage": np.nan,
                        "test_acc": np.nan,
                        "test_f1": np.nan,
                        "test_auc": np.nan
                    }
                    if args.save_wandb:
                        wandb.log(cur_result)
                    if args.display:
                        print(f"After {t + 1} iterations:")
                        print("No enough LFs. Skip training label model and end model.")

                    continue

                elif is_valid_snorkel_input(lfs):
                    label_model = get_wrench_label_model(args.label_model, verbose=False)
                    search_space = SEARCH_SPACE.get(args.label_model, None)
                else:
                    label_model = get_wrench_label_model("MV")
                    search_space = None

                # evaluate LF quality
                lf_labels = [lf.label for lf in lfs]
                gt_train_labels = np.array(train_dataset.labels)
                lf_train_stats = evaluate_lfs(gt_train_labels, L_train, np.array(lf_labels), n_class=train_dataset.n_class)
                gt_valid_labels = np.array(valid_dataset.labels)
                lf_val_stats = evaluate_lfs(gt_valid_labels, L_val, np.array(lf_labels), n_class=valid_dataset.n_class)

                if args.tune_metric == "f1":
                    metric = "f1_binary" if train_dataset.n_class == 2 else "f1_macro"
                else:
                    metric = "acc"

                if search_space is not None and args.tune_label_model:
                    train_dataset.weak_labels = L_train.tolist()
                    valid_dataset.weak_labels = L_val.tolist()
                    searched_paras = grid_search(label_model, dataset_train=train_dataset,
                                                 dataset_valid=valid_dataset,
                                                 metric=metric, direction='auto',
                                                 search_space=search_space,
                                                 n_repeats=1, n_trials=100, parallel=False)
                    label_model = get_wrench_label_model(args.label_model, **searched_paras, verbose=False)

                label_model.fit(dataset_train=train_dataset,
                                dataset_valid=valid_dataset,
                                )
                ys_tr = label_model.predict(L_train)
                ys_tr_soft = label_model.predict_proba(L_train)
                train_covered_indices = (np.max(L_train, axis=1) != -1) & (ys_tr != -1)  # indices covered by LFs
                ys_val = label_model.predict(L_val)
                valid_covered_indices = (np.max(L_val, axis=1) != -1) & (ys_val != -1)  # indices covered by LFs

                if args.default_class is None:
                    ys_tr[~train_covered_indices] = -1
                    ys_val[~valid_covered_indices] = -1
                else:
                    ys_tr[~train_covered_indices] = args.default_class
                    ys_val[~valid_covered_indices] = args.default_class
                    train_covered_indices = np.repeat(True, len(train_dataset))

                # evaluate label quality
                response_acc = accuracy_score(gt_labels, response_labels)
                train_label_stats = evaluate_labels(gt_train_labels, ys_tr, n_class=train_dataset.n_class)
                valid_label_stats = evaluate_labels(gt_valid_labels, ys_val, n_class=valid_dataset.n_class)

                # train end model
                xs_tr = train_dataset.features[train_covered_indices, :]
                ys_tr = ys_tr[train_covered_indices]
                ys_tr_soft = ys_tr_soft[train_covered_indices, :]
                if np.min(ys_tr) != np.max(ys_tr):
                    disc_model = train_disc_model(model_type=args.end_model,
                                                  xs_tr=xs_tr,
                                                  ys_tr_soft=ys_tr_soft,
                                                  ys_tr_hard=ys_tr,
                                                  valid_dataset=valid_dataset,
                                                  soft_training=args.use_soft_labels,
                                                  tune_end_model=args.tune_end_model,
                                                  tune_metric=args.tune_metric,
                                                  seed=seed)
                    # evaluate end model performance
                    test_preds = disc_model.predict(test_dataset.features)
                    gt_test_labels = np.array(test_dataset.labels)
                    test_stats = evaluate_labels(gt_test_labels, test_preds, n_class=test_dataset.n_class)
                    test_perf = evaluate_disc_model(disc_model, test_dataset)

                    valid_preds = disc_model.predict(valid_dataset.features)
                    valid_stats = evaluate_labels(gt_valid_labels, valid_preds, n_class=valid_dataset.n_class)

                else:
                    test_perf = {"acc": np.nan, "f1": np.nan, "auc": np.nan}
                    valid_stats = {}
                    test_stats = {}

                cur_result = {
                    "num_query": t+1,
                    "lf_num": len(lfs),
                    "lf_acc_avg": lf_train_stats["lf_acc_avg"],
                    "lf_cov_avg": lf_train_stats["lf_cov_avg"],
                    "response_acc": response_acc,
                    "train_precision": train_label_stats["accuracy"],
                    "train_coverage": train_label_stats["coverage"],
                    "test_acc": test_perf["acc"],
                    "test_f1": test_perf["f1"],
                    "test_auc": test_perf["auc"]
                }
                if args.early_stop:
                    if args.tune_metric == "f1":
                        valid_perf = valid_stats["f1"]
                    else:
                        valid_perf = valid_stats["acc"]
                    if valid_perf > best_valid_perf:
                        best_valid_perf = valid_perf
                        tolerance = 0
                    else:
                        tolerance += 1
                        if tolerance >= 3:
                            break

                if args.save_wandb:
                    wandb.log(cur_result)

                if args.display:
                    print(f"After {t+1} iterations:")
                    print("Train LF stats:")
                    pprint.pprint(lf_train_stats)
                    print("Valid LF stats:")
                    pprint.pprint(lf_val_stats)
                    print("Train label stats (label model output):")
                    pprint.pprint(train_label_stats)
                    print("Valid label stats (label model output):")
                    pprint.pprint(valid_label_stats)
                    print("Valid prediction stats (end model output):")
                    pprint.pprint(valid_stats)
                    print("Test prediction stats (end model output):")
                    pprint.pprint(test_stats)

        if args.save_wandb:
            wandb.run.summary["num_query"] = t+1
            wandb.run.summary["lf_num"] = len(lfs)
            wandb.run.summary["response_acc"] = response_acc
            response_freq = np.mean(np.array(response_labels) != -1)
            wandb.run.summary["response_freq"] = response_freq

            if len(lfs) > 0:
                wandb.run.summary["lf_acc_avg"] = lf_train_stats["lf_acc_avg"]
                wandb.run.summary["lf_cov_avg"] = lf_train_stats["lf_cov_avg"]
                wandb.run.summary["lf_overlap_avg"] = lf_train_stats["lf_overlap_avg"]
                wandb.run.summary["lf_conflict_avg"] = lf_train_stats["lf_conflict_avg"]
            else:
                wandb.run.summary["lf_acc_avg"] = np.nan
                wandb.run.summary["lf_cov_avg"] = np.nan
            if label_model is not None:
                wandb.run.summary["train_precision"] = train_label_stats["accuracy"]
                wandb.run.summary["train_coverage"] = train_label_stats["coverage"]
            else:
                wandb.run.summary["train_precision"] = np.nan
                wandb.run.summary["train_coverage"] = np.nan

            if disc_model is not None:
                wandb.run.summary["test_acc"] = test_perf["acc"]
                wandb.run.summary["test_f1"] = test_perf["f1"]
                wandb.run.summary["test_auc"] = test_perf["auc"]
            else:
                wandb.run.summary["test_acc"] = np.nan
                wandb.run.summary["test_f1"] = np.nan
                wandb.run.summary["test_auc"] = np.nan

            wandb.finish()
            runs_and_lfs[run] = lfs

    with open('runs_and_lfs.pkl', 'wb') as file:
        pickle.dump(runs_and_lfs, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument("--dataset-path", type=str, default="./data/wrench_data", help="dataset path")
    parser.add_argument("--dataset-name", type=str, default="youtube", help="dataset name")
    parser.add_argument("--feature-extractor", type=str, default="bert", help="feature for training end model")
    parser.add_argument("--stop-words", type=str, default=None)
    parser.add_argument("--stemming", type=str, default="porter")
    parser.add_argument("--append-cdr", action="store_true", help="append cdr snippets to original dataset")
    # sampler
    parser.add_argument("--sampler", type=str, default="passive", choices=["passive", "uncertain", "QBC", "SEU", "weighted"],
                        help="sample selector")
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L12-v2")
    parser.add_argument("--distance", type=str, default="cosine")
    parser.add_argument("--uncertain-metric", type=str, default="entropy")
    parser.add_argument("--neighbor-num", type=int, default=10, help="neighbor count in KNN")
    parser.add_argument("--alpha", type=float, default=1.0, help="trade-off factor for uncertainty. ")
    parser.add_argument("--beta", type=float, default=1.0, help="trade-off factor for class balance. ")
    parser.add_argument("--gamma", type=float, default=1.0, help="trade-off factor for distance to labeled set.")
    # data programming
    parser.add_argument("--label-model", type=str, default="Snorkel", choices=["Snorkel", "MeTaL", "MV"], help="label model used in DP paradigm")
    parser.add_argument("--use-soft-labels", action="store_true", help="set to true if use soft labels when training end model")
    parser.add_argument("--end-model", type=str, default="logistic", choices=["logistic", "mlp"], help="end model in DP paradigm")
    parser.add_argument("--default-class", type=int, default=None)
    parser.add_argument("--tune-label-model", type=bool, default=True, help="tune label model hyperparameters")
    parser.add_argument("--tune-end-model", type=bool, default=True, help="tune end model hyperparameters")
    parser.add_argument("--tune-metric", type=str, default="acc", help="evaluation metric used to tune model hyperparameters")
    # label function
    parser.add_argument("--lf-agent", type=str, default="chatgpt", choices=["chatgpt", "llama-2", "wrench"], help="agent that return candidate LFs")
    parser.add_argument("--lf-type", type=str, default="keyword", choices=["keyword", "regex"], help="LF family")
    parser.add_argument("--lf-filter", type=str, nargs="+", default=["acc", "overlap"], help="filters for LF verification")
    parser.add_argument("--lf-acc-threshold", type=float, default=0.6, help="LF accuracy threshold for verification")
    parser.add_argument("--lf-overlap-threshold", type=float, default=0.95, help="LF overlap threshold for verification")
    parser.add_argument("--max-lf-per-iter", type=int, default=100, help="Maximum LF num per interaction")
    parser.add_argument("--max-ngram", type=int, default=3, help="N-gram in keyword LF")
    # prompting method
    parser.add_argument("--lf-llm-model", type=str, default="gpt-3.5-turbo-0613")
    parser.add_argument("--example-per-class", type=int, default=1)
    parser.add_argument("--return-explanation", action="store_true")
    parser.add_argument("--example-selection", type=str, default="random", choices=["random", "neighbor"])
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1)
    parser.add_argument("--n-completion", type=int, default=1)
    # experiment
    parser.add_argument("--num-query", type=int, default=50, help="total selected samples")
    parser.add_argument("--train-iter", type=int, default=10, help="evaluation interval")
    parser.add_argument("--sleep-time", type=float, default=0, help="sleep time in seconds before each query")
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--save-wandb", action="store_true")
    args = parser.parse_args()
    main(args)


