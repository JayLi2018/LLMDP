import os
data_root = "./data/wrench_data"
datasets = [
    "youtube",
    "sms",
    "imdb",
    "yelp",
    "agnews",
    "trec",
    "chemprot",
    "cdr",
    "spouse",
    "semeval"
]

imbalanced_datasets = ["sms", "cdr", "spouse"]
label_models = ["Snorkel"]
end_models = ["logistic"]

for dataset in datasets:
    if dataset in imbalanced_datasets:
        metric = "f1"
    else:
        metric = "acc"

    for lm in label_models:
        for em in end_models:
            cmd = f"python main.py --dataset-name {dataset} --lf-agent wrench --label-model {lm} --end-model {em} " \
                  f"--tune-metric {metric} --save-wandb"
            print(cmd)
            os.system(cmd)
