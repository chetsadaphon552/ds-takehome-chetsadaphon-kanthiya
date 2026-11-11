"""
Problem 1 - Discrete Mathematics: Shareholding Networks
Part 3: Entity Resolution and Blocking Comparison Counts

ANSWERS TO PROBLEM 1.3:
a) All-pairs comparisons: n(n-1)/2 = 2,000,000 × 1,999,999 / 2 ≈ 2 trillion
b) Equal blocking (k blocks): n(n-k)/(2k)
   For k=1,000: 2,000,000(2,000,000-1,000)/(2×1,000) ≈ 2 billion
c) Convexity proof: See report/q1/convexity_proof.txt
   Function f(m)=m(m-1)/2 is strictly convex (f''(m)=1>0)
   By Jensen's inequality: Equal sizes minimize comparisons
d) Blocking misses: Records in different blocks (e.g., "Smith" vs "Smyth")
   Mitigation: Multi-pass blocking, phonetic encoding, overlapping blocks

Compares:
1. Untuned Baseline: Brute-Force All-Pairs (k = 1, no blocking).
2. Default Problem Setting: Equal Blocking with k = 1,000 blocks.
3. Candidate-pair count sensitivity over several block counts:
   - Reports candidate-pair counts only. Duplicate recall requires labelled
     duplicate pairs and is not estimated by this script.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def total_pairwise_comparisons(n: int) -> int:
    """คำนวณจำนวนคู่ทั้งหมดเมื่อไม่แบ่งบล็อก: C(n, 2)."""
    return n * (n - 1) // 2


def equal_blocking_comparisons(n: int, k: int) -> int:
    """
    Return the minimum within-block comparison count for k balanced blocks.

    When n is not divisible by k, block sizes differ by at most one.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if k <= 0:
        raise ValueError("k must be positive")

    base_size, remainder = divmod(n, k)
    larger_blocks = remainder
    smaller_blocks = k - remainder
    return (
        smaller_blocks * (base_size * (base_size - 1) // 2)
        + larger_blocks * (base_size * (base_size + 1) // 2)
    )


def run_blocking_tuning(n: int = 2_000_000):
    # Compare candidate-pair counts for several balanced block counts.
    print("=" * 80)
    print("PROBLEM 1: ENTITY RESOLUTION BLOCKING COMPARISON COUNTS")
    print(f"Total Records: n = {n:,}")
    print("=" * 80)

    # 1. Baseline: ไม่มี blocking เปรียบทุกคู่ O(n²)
    brute_force = total_pairwise_comparisons(n)
    print(f"[Untuned Baseline] k = 1 (All-Pairs Brute Force):")
    print(f"  Total Comparisons: {brute_force:,} (~{brute_force/1e12:.2f} Trillion)")

    # 2. Default k=1,000: แบ่ง record เป็น 1,000 blocks เปรียบแค่ภายใน block
    default_k = 1_000
    default_comps = equal_blocking_comparisons(n, default_k)
    default_reduction = (1 - default_comps / brute_force) * 100
    print(f"\n[Problem Default] k = {default_k:,} Equal Blocks (Size: {n//default_k:,}):")
    print(f"  Total Comparisons: {default_comps:,} (~{default_comps/1e9:.2f} Billion)")
    print(f"  Reduction Ratio  : {default_reduction:.4f}%")

    # Compare candidate-pair counts only; recall cannot be inferred without labels.
    candidate_k = [50, 100, 500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000]
    records = []

    for k in candidate_k:
        comps = equal_blocking_comparisons(n, k)
        smaller_size, remainder = divmod(n, k)
        larger_size = smaller_size + (remainder > 0)
        reduction = (1 - comps / brute_force) * 100

        records.append({
            "Block Count (k)": k,
            "Block Size Range": f"{smaller_size}-{larger_size}",
            "Comparisons": comps,
            "Reduction (%)": round(reduction, 4),
        })

    df_tuning = pd.DataFrame(records)
    print("\n" + "=" * 80)
    print("BLOCKING SENSITIVITY TABLE (candidate-pair counts only):")
    print("=" * 80)
    print(df_tuning.to_string(index=False))

    # Plot candidate-pair counts; this is not a recall or optimal-k evaluation.
    try:
        os.makedirs("report/q1", exist_ok=True)
        fig, ax1 = plt.subplots(figsize=(10, 5))

        k_vals = df_tuning["Block Count (k)"]
        comps_billions = df_tuning["Comparisons"] / 1e9

        color = "#1f77b4"
        ax1.set_xlabel("Number of Blocks (k)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Within-block Candidate Pairs (Billions)", color=color, fontsize=11, fontweight="bold")
        line1 = ax1.plot(k_vals, comps_billions, color=color, marker="o", linewidth=2.5, label="Comparisons (Billions)")
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.set_xscale("log")
        ax1.grid(True, linestyle=":", alpha=0.6)

        ax1.axvline(x=1000, color="green", linestyle=":", linewidth=2, label="Assessment example (k = 1,000)")
        ax1.legend(loc="best", fontsize=10)

        plt.title("Problem 1: Blocking Candidate-Pair Counts (Recall Not Measured)", fontsize=12, fontweight="bold")
        plt.tight_layout()
        chart_path = "report/q1/q1_blocking_tradeoff.png"
        plt.savefig(chart_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"\n[+] Blocking trade-off visualization saved to: {chart_path}")
    except Exception as e:
        print(f"[!] Plotting note: {e}")

    print("=" * 80)
    print(f"At k = {default_k:,}, blocking reduces candidate pairs by {default_reduction:.4f}%.")
    print("This is a comparison-count result only; it does not estimate duplicate recall or establish an optimal k.")
    print("=" * 80)


if __name__ == "__main__":
    run_blocking_tuning(n=2_000_000)
