from pymoo.indicators.igd import IGD
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from utils import read_score_from_path

def calculate_true_pareto_front(folder_list: list[str]) -> np.ndarray:
    full_scores = []
    
    for folder in folder_list:
        folder_path = Path(folder)
        for file_path in folder_path.rglob("samples_1~200.json"):  # recursive search
            try:
                print(f"Get from file path: {file_path}")
                with open(file_path, "r") as f:
                    data = json.load(f)
                scores = [item.get("score") for item in data if item.get("score") is not None]
                scores = [[abs(x) for x in pair] for pair in scores]
                full_scores.extend(scores)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    F_hist_np = np.array(full_scores)
    true_nd_indices = NonDominatedSorting().do(F_hist_np, only_non_dominated_front=True)
    true_pf_approx = F_hist_np[true_nd_indices]  # get true Pareto front
    
    return true_pf_approx

true_pf_approx = calculate_true_pareto_front(["logs/momcts/tsp", "logs/meoh/tsp", "logs/nsga2/tsp", "logs/momcts/tsp_trash", "logs/meoh/tsp_trash"])


def calculate_igd_from_path(json_path: str, true_pf_approx: np.ndarray) -> dict:
    print(f"Pareto front: {true_pf_approx}")
    F_hist = read_score_from_path(json_path)
    F_hist = np.array(F_hist)

    target_evals = list(range(0, 201, 10))

    igd_at_targets = {}
    metric = IGD(true_pf_approx, zero_to_one=True)

    archive = []
    for target in target_evals:
        archive.extend(F_hist[:target+1])
        nd_idx = NonDominatedSorting().do(np.array(archive), only_non_dominated_front=True)
        P = np.array(archive)[nd_idx]
        igd_at_targets[target] = metric.do(P)


    return igd_at_targets

def aggregate_igd_curves(json_paths: list[str], true_pf_approx: np.ndarray):
    """
    Compute IGD across multiple runs, return mean and std curves.
    """
    all_curves = []

    for path in json_paths:
        igd_at_targets = calculate_igd_from_path(path, true_pf_approx)
        all_curves.append(igd_at_targets)

    # Align evaluations across runs
    eval_points = sorted(set().union(*[curve.keys() for curve in all_curves]))
    all_arrays = []

    for curve in all_curves:
        arr = [curve.get(ev, np.nan) for ev in eval_points]
        all_arrays.append(arr)

    all_arrays = np.array(all_arrays)  # shape (n_runs, n_points)

    mean_curve = np.nanmean(all_arrays, axis=0)
    std_curve = np.nanstd(all_arrays, axis=0)

    return eval_points, mean_curve, std_curve


def compare_igd_curves_multi(algorithms: dict, true_pf_approx: np.ndarray):
    plt.figure(figsize=(7, 5))

    for label, paths in algorithms.items():
        evals, mean_curve, std_curve = aggregate_igd_curves(paths, true_pf_approx)

        plt.plot(evals, mean_curve, marker="o", label=label)
        plt.fill_between(evals, mean_curve-std_curve, mean_curve+std_curve, alpha=0.2)

    plt.xlabel("Function Evaluations")
    plt.ylabel("IGD")
    plt.ylim(0, 1)

    plt.title("IGD Comparison (Mean ± Std across runs)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    algorithms = {
        "MoMCTS": [
           "logs/momcts/tsp/20250913_205659/samples/samples_1~200.json"
        ],
        "MEOH": [
            "logs/meoh/tsp/tsp_v1/samples/samples_1~200.json",
            "logs/meoh/tsp/tsp_v2/samples/samples_1~200.json",
            "logs/meoh/tsp/tsp_v3/samples/samples_1~200.json"
        ],
        "NSGA2": [
            "logs/nsga2/tsp/tsp_v1/samples/samples_1~200.json",
            "logs/nsga2/tsp/tsp_v2/samples/samples_1~200.json",
            "logs/nsga2/tsp/tsp_v3/samples/samples_1~200.json"
        ]
    }

    compare_igd_curves_multi(algorithms, true_pf_approx)
