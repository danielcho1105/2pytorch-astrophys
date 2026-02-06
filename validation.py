"""
validation.py - Validate the anomaly detection results against scientific expectations.

This is where we check whether our model is producing scientifically meaningful
results. The key tests are:

1. SN2009ip MUST rank in the top 10 (it's our known precursor case)
2. Type IIn supernovae SHOULD rank higher on average than Type II
3. The anomaly scores should show meaningful spread (not all the same)
4. The ensemble models should roughly agree (low score variance)

If these tests fail, something is wrong with the data processing or model.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def validate_results(results):
    """
    Run all validation checks on the anomaly detection results.

    Args:
        results: list of dicts from compute_anomaly_scores(), sorted by score

    Returns:
        validation_report: dict with test results and summary
    """
    report = {
        "tests": [],
        "passed": 0,
        "failed": 0,
        "warnings": 0,
    }

    n_total = len(results)

    # ============================================================
    # TEST 1: SN2009ip ranking
    # This is THE critical test. SN2009ip had documented precursor
    # eruptions in 2009 and 2010. If our model doesn't flag it,
    # the anomaly detection isn't capturing what we care about.
    # ============================================================
    sn2009ip_result = None
    for r in results:
        if "2009ip" in r["name"].lower():
            sn2009ip_result = r
            break

    if sn2009ip_result:
        rank = sn2009ip_result["rank"]
        score = sn2009ip_result["anomaly_score"]
        top_10_threshold = min(10, n_total)

        test1 = {
            "name": "SN2009ip Ranking",
            "description": f"SN2009ip should rank in top {top_10_threshold}",
            "result": f"Rank {rank} out of {n_total} (score: {score:.6f})",
        }

        if rank <= top_10_threshold:
            test1["status"] = "PASSED"
            report["passed"] += 1
        elif rank <= n_total // 2:
            test1["status"] = "WARNING"
            test1["note"] = (f"Rank {rank} is above median but not top 10. "
                           "Model partially captures precursor signal.")
            report["warnings"] += 1
        else:
            test1["status"] = "FAILED"
            test1["note"] = (f"Rank {rank} is below median. Model is not "
                           "capturing precursor activity effectively.")
            report["failed"] += 1
        report["tests"].append(test1)
    else:
        report["tests"].append({
            "name": "SN2009ip Ranking",
            "status": "SKIPPED",
            "note": "SN2009ip not found in results (data fetch may have failed)",
        })

    # ============================================================
    # TEST 2: Type IIn vs Type II comparison
    # Type IIn supernovae interact with circumstellar material (CSM)
    # from pre-explosion mass loss. They should show more unusual
    # light curve features than normal Type II.
    # ============================================================
    iin_scores = [r["anomaly_score"] for r in results if r["type"] == "IIn"]
    ii_scores = [r["anomaly_score"] for r in results if r["type"] == "II"]

    if len(iin_scores) >= 2 and len(ii_scores) >= 2:
        iin_median = np.median(iin_scores)
        ii_median = np.median(ii_scores)

        test2 = {
            "name": "Type IIn vs Type II",
            "description": "Type IIn should have higher median anomaly score than Type II",
            "result": f"IIn median: {iin_median:.6f}, II median: {ii_median:.6f}",
        }

        if iin_median > ii_median:
            test2["status"] = "PASSED"
            test2["note"] = (f"IIn is {iin_median/ii_median:.1f}x higher than II. "
                           "Model correctly identifies IIn as more unusual.")
            report["passed"] += 1
        else:
            test2["status"] = "WARNING"
            test2["note"] = ("Type II scoring higher than IIn. This could mean: "
                           "(1) data quality issues, (2) not enough IIn samples, "
                           "or (3) the features don't capture CSM interaction well.")
            report["warnings"] += 1
        report["tests"].append(test2)
    else:
        report["tests"].append({
            "name": "Type IIn vs Type II",
            "status": "SKIPPED",
            "note": f"Not enough samples (IIn: {len(iin_scores)}, II: {len(ii_scores)})",
        })

    # ============================================================
    # TEST 3: Score spread (meaningful discrimination)
    # If all scores are nearly the same, the model isn't learning anything.
    # We want to see meaningful spread.
    # ============================================================
    all_scores = [r["anomaly_score"] for r in results]
    score_std = np.std(all_scores)
    score_mean = np.mean(all_scores)

    test3 = {
        "name": "Score Spread",
        "description": "Anomaly scores should show meaningful variation",
        "result": f"Mean: {score_mean:.6f}, Std: {score_std:.6f}, "
                  f"CV: {score_std/score_mean:.3f}" if score_mean > 0
                  else f"Mean: {score_mean:.6f}, Std: {score_std:.6f}",
    }

    cv = score_std / (score_mean + 1e-10)
    if cv > 0.3:
        test3["status"] = "PASSED"
        test3["note"] = "Good score spread - model is discriminating between objects."
        report["passed"] += 1
    elif cv > 0.1:
        test3["status"] = "WARNING"
        test3["note"] = "Moderate spread. Model is partially discriminating."
        report["warnings"] += 1
    else:
        test3["status"] = "FAILED"
        test3["note"] = "Very low spread. Model may not be learning useful patterns."
        report["failed"] += 1
    report["tests"].append(test3)

    # ============================================================
    # TEST 4: Ensemble consistency
    # The individual models should roughly agree on rankings.
    # High score_std relative to anomaly_score = inconsistent models.
    # ============================================================
    avg_relative_std = np.mean([
        r["score_std"] / (r["anomaly_score"] + 1e-10) for r in results
    ])

    test4 = {
        "name": "Ensemble Consistency",
        "description": "Models in ensemble should roughly agree on scores",
        "result": f"Average relative score std: {avg_relative_std:.3f}",
    }

    if avg_relative_std < 0.5:
        test4["status"] = "PASSED"
        test4["note"] = "Good consistency across ensemble models."
        report["passed"] += 1
    elif avg_relative_std < 1.0:
        test4["status"] = "WARNING"
        test4["note"] = "Moderate disagreement between models."
        report["warnings"] += 1
    else:
        test4["status"] = "FAILED"
        test4["note"] = "High disagreement. Consider more training or different architecture."
        report["failed"] += 1
    report["tests"].append(test4)

    # ============================================================
    # TEST 5: Top-10 composition
    # We expect the top 10 to be enriched in Type IIn.
    # ============================================================
    top_10 = results[:min(10, n_total)]
    top_10_types = [r["type"] for r in top_10]
    n_iin_top10 = top_10_types.count("IIn")
    total_iin = sum(1 for r in results if r["type"] == "IIn")

    test5 = {
        "name": "Top-10 Enrichment",
        "description": "Top 10 should be enriched in Type IIn relative to the full sample",
        "result": f"Top 10 has {n_iin_top10} IIn out of {len(top_10)} "
                  f"(vs {total_iin} IIn out of {n_total} total)",
    }

    # Expected fraction of IIn in random draw
    if n_total > 0 and total_iin > 0:
        expected_frac = total_iin / n_total
        observed_frac = n_iin_top10 / len(top_10) if len(top_10) > 0 else 0

        if observed_frac > expected_frac:
            test5["status"] = "PASSED"
            test5["note"] = (f"IIn enrichment factor: {observed_frac/expected_frac:.1f}x. "
                           "Type IIn are overrepresented in the top anomalies.")
            report["passed"] += 1
        else:
            test5["status"] = "WARNING"
            test5["note"] = "No IIn enrichment in top 10."
            report["warnings"] += 1
    else:
        test5["status"] = "SKIPPED"
        test5["note"] = "Not enough data for enrichment test"
    report["tests"].append(test5)

    return report


def print_report(report):
    """Print a formatted validation report."""
    print("\n" + "=" * 70)
    print("VALIDATION REPORT")
    print("=" * 70)

    for test in report["tests"]:
        status_symbol = {
            "PASSED": "[PASS]",
            "FAILED": "[FAIL]",
            "WARNING": "[WARN]",
            "SKIPPED": "[SKIP]",
        }.get(test["status"], "[????]")

        print(f"\n{status_symbol} {test['name']}")
        print(f"  Description: {test.get('description', 'N/A')}")
        print(f"  Result: {test.get('result', 'N/A')}")
        if "note" in test:
            print(f"  Note: {test['note']}")

    print("\n" + "-" * 70)
    print(f"Summary: {report['passed']} passed, {report['warnings']} warnings, "
          f"{report['failed']} failed")
    print("=" * 70)


def print_watchlist(results, top_n=None):
    """Print the ranked watchlist in a clean format."""
    if top_n is None:
        top_n = len(results)

    print("\n" + "=" * 70)
    print("SUPERNOVA PRECURSOR WATCHLIST")
    print("Ranked by anomaly score (higher = more unusual = higher priority)")
    print("=" * 70)
    print(f"{'Rank':>4} {'Name':>15} {'Type':>5} {'Score':>12} {'Std':>10} {'Flag':>10}")
    print("-" * 70)

    mean_score = np.mean([r["anomaly_score"] for r in results])
    std_score = np.std([r["anomaly_score"] for r in results])

    for r in results[:top_n]:
        flag = ""
        if r["anomaly_score"] > mean_score + 2 * std_score:
            flag = "*** HIGH"
        elif r["anomaly_score"] > mean_score + std_score:
            flag = "** ELEVATED"
        elif r["anomaly_score"] > mean_score:
            flag = "* ABOVE AVG"

        # Special marker for SN2009ip
        name_str = r["name"]
        if "2009ip" in r["name"].lower():
            name_str += " (!)"

        print(f"{r['rank']:>4} {name_str:>15} {r['type']:>5} "
              f"{r['anomaly_score']:>12.6f} {r['score_std']:>10.6f} {flag:>10}")

    print("-" * 70)
    print(f"Mean score: {mean_score:.6f}, Std: {std_score:.6f}")
    print(f"Objects above mean + 2*std: "
          f"{sum(1 for r in results if r['anomaly_score'] > mean_score + 2 * std_score)}")
    print("=" * 70)
