import argparse
import json
from pathlib import Path
from typing import List, Dict, Union

import pandas as pd

RELATION_TYPE = {
    "awardWonBy": "string",
    "hasCapacity": "numeric",
    "hasArea": "numeric",
    "countryLandBordersCountry": "string",
    "personHasCityOfDeath": "string",
    "companyTradesAtStockExchange": "string",
    "seriesHasNumberOfEpisodes": "numeric",
}


def read_file(file_path: Union[str, Path]) -> List[Dict]:
    """
    Detect and read the input file (CSV or JSONL).
    Returns a list of dictionaries representing the rows.
    """
    file_path = Path(file_path)
    if file_path.suffix == ".jsonl":
        with open(file_path, "r") as f:
            rows = [json.loads(line) for line in f]
    elif file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
        rows = df.to_dict(orient="records")
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    return rows


def true_positives(preds: List, gts: List, rel: str, rel_type: str, tolerance: float = 0.05) -> int:
    if rel_type == "numeric":
        return true_positives_numeric(preds, gts, tolerance)
    elif rel_type == "string":
        return sum(1 for pred in preds if pred in gts)
    else:
        raise ValueError(f"Unknown relation type: {rel_type}")


def precision(preds: List[str], gts: List[str]) -> float:
    try:
        # When nothing is predicted, precision = 1
        # irrespective of the ground truth value
        if len(preds) == 0:
            return 1
        # When the predictions are not empty
        return min(true_positives(preds, gts) / len(preds), 1.0)
    except TypeError:
        return 0.0


def recall(preds: List[str], gts: List[str]) -> float:
    try:
        # When ground truth is empty return 1
        # even if there are predictions (edge case)
        if len(gts) == 0:
            return 1.0
        # When the ground truth is not empty
        return min(true_positives(preds, gts) / len(gts), 1.0)
    except TypeError:
        return 0.0


def f1_score(p: float, r: float) -> float:
    try:
        return (2 * p * r) / (p + r)
    except ZeroDivisionError:
        return 0.0


def rows_to_dict(rows: List[Dict]) -> Dict:
    """Index the ground truth/prediction rows by subject entity and relation."""
    return {
        (r["SubjectEntity"], r["Relation"]): list(set(eval(r["ObjectEntitiesID"])))
        for r in rows
    }


def rows_to_dict_string(rows: List[Dict]) -> Dict:
    """Index the ground truth/prediction rows by subject entity and relation, using ObjectEntities (strings)."""
    return {
        (r["SubjectEntity"], r["Relation"]): list(set(eval(r["ObjectEntities"])))
        for r in rows
    }


def true_positives_numeric(preds: List, gts: List, tolerance: float = 0.05) -> int:
    tp = 0
    used = set()
    for pred in preds:
        try:
            pred_num = float(pred)
            for j, gt in enumerate(gts):
                if j in used:
                    continue
                gt_num = float(gt)
                if abs(pred_num - gt_num) / max(abs(gt_num), 1e-8) <= tolerance:
                    tp += 1
                    used.add(j)
                    break
        except Exception:
            continue
    return tp


def evaluate_per_sr_pair(pred_rows, gt_rows, rel_types=RELATION_TYPE, tolerance=0.05) -> List[Dict[str, float]]:
    pred_dict = rows_to_dict(pred_rows)
    gt_dict = rows_to_dict(gt_rows)
    results = []
    for subj, rel in gt_dict:
        gts = gt_dict[(subj, rel)]
        preds = pred_dict.get((subj, rel), [])
        rel_type = rel_types.get(rel, "string")
        tp = true_positives(preds, gts, rel=rel, rel_type=rel_type, tolerance=tolerance)
        p = tp / len(preds) if preds else 1.0
        r = tp / len(gts) if gts else 1.0
        f1 = f1_score(p, r)
        results.append({
            "SubjectEntity": subj,
            "Relation": rel,
            "p": p,
            "r": r,
            "f1": f1,
            "tp": tp,
            "total_pred": len(preds),
            "total_gt": len(gts),
        })
    return sorted(results, key=lambda x: (x["Relation"], x["SubjectEntity"]))


def evaluate_per_sr_pair_string(pred_rows, gt_rows, rel_types=RELATION_TYPE, tolerance=0.05):
    pred_dict = rows_to_dict_string(pred_rows)
    gt_dict = rows_to_dict_string(gt_rows)
    results = []
    for subj, rel in gt_dict:
        gts = gt_dict[(subj, rel)]
        preds = pred_dict.get((subj, rel), [])
        rel_type = rel_types.get(rel, "string")
        tp = true_positives(preds, gts, rel=rel, rel_type=rel_type, tolerance=tolerance)
        p = tp / len(preds) if preds else 1.0
        r = tp / len(gts) if gts else 1.0
        f1 = f1_score(p, r)
        results.append({
            "SubjectEntity": subj,
            "Relation": rel,
            "p": p,
            "r": r,
            "f1": f1,
            "tp": tp,
            "total_pred": len(preds),
            "total_gt": len(gts),
        })
    return sorted(results, key=lambda x: (x["Relation"], x["SubjectEntity"]))


def macro_average_per_relation(scores_per_sr: List[Dict[str, float]]) -> dict:
    """Compute the macro average scores per relation"""
    scores = {}
    for r in scores_per_sr:
        if r["Relation"] not in scores:
            scores[r["Relation"]] = []
        scores[r["Relation"]].append({
            "p": r["p"],
            "r": r["r"],
            "f1": r["f1"],
        })

    macro_averages = {}
    for rel in scores:
        macro_averages[rel] = {
            "macro-p": sum([x["p"] for x in scores[rel]]) / len(scores[rel]),
            "macro-r": sum([x["r"] for x in scores[rel]]) / len(scores[rel]),
            "macro-f1": sum([x["f1"] for x in scores[rel]]) / len(scores[rel]),
        }

    # Macro average for all relations
    all_rel_macro_p = sum([x["p"] for x in scores_per_sr]) / len(scores_per_sr)
    all_rel_macro_r = sum([x["r"] for x in scores_per_sr]) / len(scores_per_sr)
    all_rel_macro_f1 = sum([x["f1"] for x in scores_per_sr]) / len(
        scores_per_sr)

    macro_averages["*** All Relations ***"] = {
        "macro-p": all_rel_macro_p,
        "macro-r": all_rel_macro_r,
        "macro-f1": all_rel_macro_f1,
    }

    return macro_averages


def micro_average_per_relation(scores_per_sr: List[Dict[str, float]]) -> dict:
    """Compute the micro average scores per relation"""
    scores = {}
    for r in scores_per_sr:
        if r["Relation"] not in scores:
            scores[r["Relation"]] = {
                "tp": 0,
                "total_pred": 0,
                "total_gt": 0,
            }
        scores[r["Relation"]]["tp"] += r["tp"]
        scores[r["Relation"]]["total_pred"] += r["total_pred"]
        scores[r["Relation"]]["total_gt"] += r["total_gt"]

    micro_averages = {}
    for rel in scores:
        micro_p = scores[rel]["tp"] / scores[rel]["total_pred"] if scores[rel][
                                                                       "total_pred"] > 0 else 1.0
        micro_r = scores[rel]["tp"] / scores[rel]["total_gt"] if scores[rel][
                                                                     "total_gt"] > 0 else 1.0

        micro_averages[rel] = {
            "micro-p": micro_p,
            "micro-r": micro_r,
            "micro-f1": f1_score(micro_p, micro_r),
        }

    # Micro average for all relations
    total_tp = sum([x["tp"] for x in scores.values()])
    total_pred = sum([x["total_pred"] for x in scores.values()])
    total_gt = sum([x["total_gt"] for x in scores.values()])

    all_rel_micro_p = total_tp / total_pred if total_pred > 0 else 1.0
    all_rel_micro_r = total_tp / total_gt if total_gt > 0 else 1.0

    micro_averages["*** All Relations ***"] = {
        "micro-p": all_rel_micro_p,
        "micro-r": all_rel_micro_r,
        "micro-f1": f1_score(all_rel_micro_p, all_rel_micro_r),
    }

    return micro_averages


def prediction_statistics(scores_per_sr: List[Dict[str, float]]) -> dict:
    """Get the average numbers of predictions and the numbers of empty predictions per relation."""
    stats = {}
    for r in scores_per_sr:
        if r["Relation"] not in stats:
            stats[r["Relation"]] = {
                "num_sr_pairs": 0,
                "total_pred": 0,
                "empty_pred": 0,
            }
        stats[r["Relation"]]["num_sr_pairs"] += 1
        stats[r["Relation"]]["total_pred"] += r["total_pred"]
        if r["total_pred"] == 0:
            stats[r["Relation"]]["empty_pred"] += 1

    final_stats = {}
    for rel in stats:
        final_stats[rel] = {
            "avg. #preds": stats[rel]["total_pred"] / stats[rel][
                "num_sr_pairs"],
            "#empty preds": stats[rel]["empty_pred"],
        }

    # Average numbers of predictions and the numbers of empty predictions for all relations
    total_sr_pairs = len(scores_per_sr)
    total_preds = sum([x["total_pred"] for x in stats.values()])
    total_empty_preds = sum([x["empty_pred"] for x in stats.values()])

    final_stats["*** All Relations ***"] = {
        "avg. #preds": total_preds / total_sr_pairs,
        "#empty preds": total_empty_preds,
    }

    return final_stats


def evaluate(predictions, ground_truth, mode="ids", rel_types=RELATION_TYPE, tolerance=0.05, verbose=True):
    """
    Evaluate predictions against ground truth.
    mode: 'ids' (default) for QID-based, 'string' for string/numeric-based
    verbose: Whether to print evaluation results
    """
    pred_rows = read_file(predictions)
    gt_rows = read_file(ground_truth)
    if mode == "string":
        scores_per_sr_pair = evaluate_per_sr_pair_string(pred_rows, gt_rows, rel_types, tolerance)
    else:
        scores_per_sr_pair = evaluate_per_sr_pair(pred_rows, gt_rows, rel_types, tolerance)
    macro_per_relation = macro_average_per_relation(scores_per_sr_pair)
    macro_df = pd.DataFrame(macro_per_relation).transpose().round(3)
    micro_per_relation = micro_average_per_relation(scores_per_sr_pair)
    micro_df = pd.DataFrame(micro_per_relation).transpose().round(3)
    stats = prediction_statistics(scores_per_sr_pair)
    stats_df = pd.DataFrame(stats).transpose().round(3)
    stats_df["#empty preds"] = stats_df["#empty preds"].astype(int)
    results = pd.concat([macro_df, micro_df, stats_df], axis=1)
    if verbose:
        print(results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluating predictions.")
    parser.add_argument("predictions", type=str, help="Path to predictions file (CSV or JSONL).")
    parser.add_argument("ground_truth", type=str, help="Path to ground truth file (CSV or JSONL).")
    parser.add_argument("--mode", type=str, default="ids", choices=["ids", "string"], help="Evaluation mode: 'ids' (QID-based) or 'string' (string/numeric-based)")
    parser.add_argument("--tolerance", type=float, default=None, help="Relative error tolerance for numeric relations (default: 0 for ids mode, 0.05 for string mode)")
    args = parser.parse_args()
    
    # Set default tolerance based on mode if not explicitly provided
    if args.tolerance is None:
        args.tolerance = 0.05 if args.mode == "string" else 1e-8
    
    evaluate(args.predictions, args.ground_truth, mode=args.mode, rel_types=RELATION_TYPE, tolerance=args.tolerance)