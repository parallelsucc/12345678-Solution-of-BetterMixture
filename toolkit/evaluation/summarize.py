import argparse
import json
import sys
from pathlib import Path
from tabulate import tabulate

# stage1 dev baichuan2-7b-base
BASELINE = {
    "arc_challenge": 40.0000,
    "hellaswag": 80.0000,
    "truthfulqa_mc": 47.9730,
    "hendrycksTest-*": 54.4098,
    "cmmlu-*": 59.2946,
    "gsm8k": 20.0000,
    "scrolls_summscreenfd": 2.7047,
}


def get_preferred_metric(val, preferred_metrics=["acc_norm", "acc", "mc2", "rougeL"]):
    for metric in preferred_metrics:
        if metric in val:
            return val[metric] * 100.0
    return 0


def get_task_score(path):
    try:
        data = json.loads(path.read_text())
        sub_scores = [get_preferred_metric(val) for val in data["results"].values()]
        return sum(sub_scores) / len(sub_scores)
    except:
        return 0.0


def parse_results(output_path, BASELINE):
    scores = {}
    ratios = {}
    for task, baseline in BASELINE.items():
        path = Path(output_path, f"{task}.json")
        score = get_task_score(path)
        scores[task] = score
        ratio = score / baseline if baseline else 0.0
        ratios[task] = ratio
    return scores, ratios


def make_table(**kwargs):
    def calculate_mean(values):
        try:
            return sum(values) / len(values)
        except (TypeError, ZeroDivisionError):
            return "mean"

    list_lengths = [len(value) for value in kwargs.values() if isinstance(value, list)]
    assert len(list_lengths) == len(kwargs), "All arguments must be lists."
    assert len(set(list_lengths)) == 1, "All lists must have the same length."

    result = {
        key: value + [None, calculate_mean(value)] for key, value in kwargs.items()
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    scores, ratios = parse_results(args.output_path, BASELINE)
    tasks = list(BASELINE.keys())
    data = make_table(
        task=tasks,
        baseline=[BASELINE[t] for t in tasks],
        score=[scores[t] for t in tasks],
        ratio=[ratios[t] for t in tasks],
    )

    print(
        tabulate(
            data,
            headers="keys",
            colalign=("left", "decimal"),
            tablefmt="github",
            floatfmt=".4f",
        )
    )
