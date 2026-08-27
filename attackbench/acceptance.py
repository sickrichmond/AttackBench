"""
Acceptance run: does this code reproduce the AttackBench protocol?

Runs one or more attacks under the paper's conditions (untargeted, fixed query budget,
best-iterate distances) and prints a table with the same columns as Table I of the paper
— Attack | Library | ASR | GO/LO | #F | #B | t(s) — so it can be read side by side with
the published results and the leaderboard at https://attackbench.github.io.

No expected values are hardcoded here on purpose: transcribing them by hand is how a
benchmark ends up asserting against the wrong numbers. Compare the printed table with
the paper yourself.

Two ways to get the reference a* (the lower envelope local optimality is measured
against), chosen with --reference:

  ensemble  the attacks run in this session, sample-wise minimum. Self-contained and
            needs no credentials, but the scores are relative to THIS set of attacks:
            the best attack in the set scores ~1.0 by construction. Use it to check
            ranking and behaviour, not to compare with the paper's absolute numbers.

  wandb     the optimal distances published on W&B, matched per-sample by SHA-512 hash.
            This is the paper's definition (a* ensembles every attack in the benchmark)
            and the only mode whose scores are comparable with the published ones.
            NOTE: artifacts uploaded before v2.0 hold last-iterate distances, not d*,
            so they under-state the envelope. Repopulate them first — see
            attackbench.update_optimal_distances().

Examples
--------
# paper setup for CIFAR-10 (full test set, batch 128, Q=2000)
python scripts/paper_acceptance.py --model standard --dataset cifar10 \
    --threat-model l2 --attacks original:fmn adv_lib:fmn adv_lib:ddn \
    --reference ensemble

# against the published envelope, on a subset
python scripts/paper_acceptance.py --model carmon_2019 --dataset cifar10 \
    --threat-model linf --attacks original:apgd art:pgd --n-samples 1000 \
    --reference wandb

# quick end-to-end smoke test on CPU (small model, small data)
python scripts/paper_acceptance.py --model mnist_smallcnn --dataset mnist \
    --threat-model l2 --attacks preconfigured:fmn preconfigured:deepfool \
    --n-samples 64 --batch-size 32 --device cpu
"""

import argparse
import statistics
import sys
import time
import warnings

import numpy as np
import torch

import attackbench
from attackbench import DEFAULT_QUERY_BUDGET


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--model",
        required=True,
        help="model key from attackbench.models.registry.MODEL_CONFIGS, "
        "lowercase (e.g. standard, carmon_2019, mnist_smallcnn)",
    )
    p.add_argument("--dataset", required=True, choices=["cifar10", "imagenet", "mnist"])
    p.add_argument("--threat-model", required=True, choices=["l0", "l1", "l2", "linf"])
    p.add_argument(
        "--attacks",
        required=True,
        nargs="+",
        metavar="LIB:NAME",
        help="attacks as library:name (e.g. original:fmn art:pgd). Use "
        "'preconfigured:<name>' for the ready-made ones",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="number of samples (default: the whole evaluation set)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="128 for CIFAR-10 and 32 for ImageNet in the paper",
    )
    p.add_argument("--seed", type=int, default=0, help="subset selection seed")
    p.add_argument(
        "--query-budget",
        type=int,
        default=DEFAULT_QUERY_BUDGET,
        help="max forward+backward propagations per sample (0 = unlimited)",
    )
    p.add_argument("--reference", choices=["ensemble", "wandb"], default="ensemble")
    p.add_argument(
        "--minimum-asr", type=float, default=None,
        help="fail the acceptance run if any attack has a lower ASR percentage",
    )
    p.add_argument("--device", default=None, help="cuda, cuda:1, cpu (default: auto)")
    p.add_argument("--data-root", default="data")
    p.add_argument("--cache-dir", default="./cache")
    return p.parse_args(argv)


def build_attack(spec: str, threat_model: str):
    """'lib:name' -> (label, callable)."""
    if ":" not in spec:
        raise SystemExit(f"--attacks entries must be 'library:name', got '{spec}'")
    lib, name = spec.split(":", 1)

    if lib == "preconfigured":
        from attackbench import attacks as preconfigured

        try:
            return f"{name}-original", getattr(preconfigured, name)
        except AttributeError:
            raise SystemExit(f"unknown preconfigured attack '{name}'")

    attack = attackbench.get_attack(lib=lib, attack=name, threat_model=threat_model)
    return f"{name}-{lib}", attack


def summarise(label, results, threat_model):
    """One row of the report, in the shape of Table I plus the d* diagnostics."""
    distances = np.asarray(results["distances"][threat_model], dtype=float)
    final = np.asarray(results["final_distances"][threat_model], dtype=float)
    solved = np.isfinite(distances)
    forwards = np.asarray(results["num_forwards"])
    backwards = np.asarray(results["num_backwards"])

    # how much the last iterate would have under-stated the attack (the pre-2.0 bug)
    both = solved & np.isfinite(final) & (distances > 0)
    gap = (
        float(np.mean((final[both] - distances[both]) / distances[both]) * 100)
        if both.any()
        else 0.0
    )

    return {
        "attack": label,
        "ASR": 100.0 * float(np.mean(results["adv_success"])),
        "clean_acc": 100.0 * float(np.mean(results["correct"])),
        "median_d": (
            float(np.median(distances[solved])) if solved.any() else float("nan")
        ),
        "mean_d": float(np.mean(distances[solved])) if solved.any() else float("nan"),
        "gap_%": gap,
        "#F": int(forwards.max()) if len(forwards) else 0,
        "#B": int(backwards.max()) if len(backwards) else 0,
        "t(s)": float(sum(results["times"])),
        "max_queries": int((forwards + backwards).max()) if len(forwards) else 0,
        "failed_batches": int(sum(results["batch_failures"])),
        "box_violations": int(sum(results["box_failures"])),
    }


def print_table(rows, optimality):
    header = (
        f"{'Attack':<26} {'ASR':>6} {'LO':>7} {'median d*':>11} {'mean d*':>10} "
        f"{'d*/last gap':>12} {'#F':>6} {'#B':>6} {'t(s)':>8}"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        lo = optimality.get(row["attack"])
        lo_str = f"{lo:.4f}" if lo is not None and not np.isnan(lo) else "  n/a"
        print(
            f"{row['attack']:<26} {row['ASR']:>5.1f}% {lo_str:>7} {row['median_d']:>11.5f} "
            f"{row['mean_d']:>10.5f} {row['gap_%']:>11.1f}% {row['#F']:>6} {row['#B']:>6} "
            f"{row['t(s)']:>8.1f}"
        )


def main(argv=None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device) if args.device else None
    budget = args.query_budget or None

    print(
        f"model={args.model} dataset={args.dataset} threat_model={args.threat_model} "
        f'n_samples={args.n_samples or "all"} batch_size={args.batch_size} '
        f"query_budget={budget} seed={args.seed} reference={args.reference}"
    )

    loader = attackbench.get_loader(
        dataset=args.dataset,
        batch_size=args.batch_size,
        num_samples=args.n_samples,
        seed=args.seed,
        root=args.data_root,
    )
    model = attackbench.get_model(args.model)

    rows, results_list, labels = [], [], []
    for spec in args.attacks:
        label, attack = build_attack(spec, args.threat_model)
        print(f"\n=== {label}")
        started = time.perf_counter()
        results = attackbench.run_attack(
            model,
            loader,
            attack,
            args.threat_model,
            device=device,
            query_budget=budget,
            use_cached=False,
        )
        row = summarise(label, results, args.threat_model)
        row["wall(s)"] = time.perf_counter() - started
        rows.append(row)
        results_list.append(results)
        labels.append(label)

    # ── local optimality against the chosen reference ────────────────────────
    optimality = {}
    if args.reference == "wandb":
        for label, results in zip(labels, results_list):
            try:
                out = attackbench.compute_local_optimality(
                    attack_results=results,
                    threat_model=args.threat_model,
                    dataset=args.dataset,
                    model_name=results["metadata"].get("model_name"),
                    use_wandb=True,
                    cache_dir=args.cache_dir,
                )
                optimality[label] = out["optimality"]
            except Exception as e:
                print(f"  optimality unavailable for {label}: {e}")
    elif len(results_list) >= 2:
        comparison = attackbench.compare_attacks_optimality(
            results_list, threat_model=args.threat_model, attack_names=labels
        )
        optimality = comparison["optimality_scores"]
    else:
        print(
            "\nnote: --reference ensemble needs at least 2 attacks to build an envelope"
        )

    print_table(rows, optimality)

    # ── protocol violations worth stopping for ───────────────────────────────
    problems = []
    for row in rows:
        if budget and row["max_queries"] > budget:
            problems.append(
                f"{row['attack']}: spent {row['max_queries']} > budget {budget}"
            )
        if row["failed_batches"]:
            problems.append(
                f"{row['attack']}: {row['failed_batches']} batch(es) raised"
            )
        if row["box_violations"]:
            problems.append(
                f"{row['attack']}: {row['box_violations']} sample(s) left [0,1]"
            )
        if args.minimum_asr is not None and row["ASR"] < args.minimum_asr:
            problems.append(
                f"{row['attack']}: ASR {row['ASR']:.1f}% < required "
                f"{args.minimum_asr:.1f}%"
            )

    clean = {round(row["clean_acc"], 2) for row in rows}
    print(
        f"\nclean accuracy: {', '.join(f'{c}%' for c in sorted(clean))} "
        f"(compare with the model's published accuracy)"
    )
    if len(clean) > 1:
        problems.append(
            f"clean accuracy differs between runs ({clean}) — the model or "
            f"the samples changed mid-run"
        )

    if problems:
        print("\nPROTOCOL WARNINGS")
        for p in problems:
            print("  - " + p)
    else:
        print("\nno protocol violations detected")

    if args.reference == "ensemble":
        print(
            "\nScores are relative to the attacks run here: an attack reaches 1.0 only by "
            "matching the\nenvelope on (nearly) every sample, so with a small set they can "
            "all sit below 1.\nFor numbers comparable with the paper, use --reference wandb "
            "against a repopulated envelope."
        )
    return 1 if problems else 0


if __name__ == "__main__":
    warnings.simplefilter("default")
    sys.exit(main())
