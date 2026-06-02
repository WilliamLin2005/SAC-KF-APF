"""Export C-vs-other complex ablation comparison figures."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

from envs.complex_nav_env import ComplexNavEnv
from evaluate.compare_complex_ab import EvalResult, evaluate_condition, summarize
from evaluate.compare_complex_pair import (
    CONDITIONS,
    plot_action_comparison,
    plot_distance_comparison,
    plot_path_comparison,
    plot_reward_comparison,
)
from evaluate.complex_baselines import evaluate_baseline


GROUP_DIR = Path("outputs/ablations/group1_complex")
DEFAULT_OUT_DIR = Path("outputs/ablations/path_compare_fig")


@dataclass(frozen=True)
class GeneratedPairSpec:
    key: str
    label: str
    baseline: str
    model_path: Path
    use_kf: bool
    description: str


GENERATED_PAIRS = (
    GeneratedPairSpec(
        key="D",
        label="D action penalty",
        baseline="action_delta_penalty",
        model_path=GROUP_DIR / "D_action_penalty.zip",
        use_kf=False,
        description="SAC train no KF with raw action-delta reward penalty.",
    ),
    GeneratedPairSpec(
        key="E",
        label="E low-pass eval-only",
        baseline="lowpass_eval_only",
        model_path=GROUP_DIR / "A_sac_train_no_kf.zip",
        use_kf=False,
        description="A no-KF policy evaluated with low-pass action filtering.",
    ),
    GeneratedPairSpec(
        key="F",
        label="F gSDE",
        baseline="gsde",
        model_path=GROUP_DIR / "F_gsde.zip",
        use_kf=False,
        description="SAC train no KF with gSDE exploration.",
    ),
    GeneratedPairSpec(
        key="G",
        label="G KF no aug",
        baseline="kf_no_aug",
        model_path=GROUP_DIR / "G_kf_no_aug.zip",
        use_kf=True,
        description="KF-in-loop training/evaluation without prev-exec-action observation augmentation.",
    ),
)


COPY_FIGURES = (
    (
        GROUP_DIR / "AC_compare" / "path_ac_complex.png",
        "A_vs_C_path_complex.png",
    ),
    (
        GROUP_DIR / "AC_compare" / "reward_curve_ac_complex.png",
        "A_vs_C_reward_curve_complex.png",
    ),
    (
        GROUP_DIR / "AC_compare" / "step_distance_ac_complex.png",
        "A_vs_C_step_distance_complex.png",
    ),
    (
        GROUP_DIR / "AC_compare" / "action_smoothing_ac_complex.png",
        "A_vs_C_action_smoothing_complex.png",
    ),
    (
        GROUP_DIR / "BC_compare" / "path_bc_complex.png",
        "B_vs_C_path_complex.png",
    ),
    (
        GROUP_DIR / "BC_compare" / "reward_curve_bc_complex.png",
        "B_vs_C_reward_curve_complex.png",
    ),
    (
        GROUP_DIR / "BC_compare" / "step_distance_bc_complex.png",
        "B_vs_C_step_distance_complex.png",
    ),
    (
        GROUP_DIR / "BC_compare" / "action_smoothing_bc_complex.png",
        "B_vs_C_action_smoothing_complex.png",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export C-vs-other complex ablation figures.")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", type=int, default=0, choices=[0, 1])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--action-penalty-weight", type=float, default=0.2)
    parser.add_argument("--lowpass-alpha", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    copied = copy_existing_figures(out_dir)
    generated = generate_baseline_vs_c_figures(args, out_dir)
    write_readme(args, out_dir, copied=copied, generated=generated)

    png_count = len(list(out_dir.glob("*.png")))
    print(f"\nSaved C comparison figures to: {out_dir}")
    print(f"PNG files in output directory: {png_count}")


def copy_existing_figures(out_dir: Path) -> list[str]:
    copied = []
    for source, destination_name in COPY_FIGURES:
        if not source.exists():
            raise FileNotFoundError(f"Existing comparison figure not found: {source}")
        destination = out_dir / destination_name
        shutil.copy2(source, destination)
        copied.append(destination.name)
        print(f"Copied {source} -> {destination}")
    return copied


def generate_baseline_vs_c_figures(args: argparse.Namespace, out_dir: Path) -> list[str]:
    c_spec = CONDITIONS["C"]
    c_model_path = Path(c_spec.model_path)
    if not c_model_path.exists():
        raise FileNotFoundError(f"C model file not found: {c_model_path}")

    c_result = evaluate_condition(
        label=c_spec.label,
        use_kf=c_spec.use_kf,
        model_path=c_model_path,
        episodes=args.episodes,
        seed=args.seed,
        deterministic=bool(args.deterministic),
        device=args.device,
    )

    generated = []
    for pair_spec in GENERATED_PAIRS:
        if not pair_spec.model_path.exists():
            raise FileNotFoundError(f"{pair_spec.key} model file not found: {pair_spec.model_path}")

        left_result = evaluate_generated_pair_condition(args, pair_spec)
        pair_label = f"{pair_spec.key} vs C"
        prefix = f"{pair_spec.key}_vs_C"

        plot_env = ComplexNavEnv(use_kf=False, seed=args.seed)
        try:
            path_file = out_dir / f"{prefix}_path_complex.png"
            reward_file = out_dir / f"{prefix}_reward_curve_complex.png"
            distance_file = out_dir / f"{prefix}_step_distance_complex.png"
            action_file = out_dir / f"{prefix}_action_smoothing_complex.png"

            plot_path_comparison(plot_env, left_result, c_result, path_file, pair_label=pair_label)
            plot_reward_comparison(left_result, c_result, reward_file, pair_label=pair_label)
            plot_distance_comparison(left_result, c_result, distance_file, pair_label=pair_label)
            plot_action_comparison(left_result, c_result, action_file, pair_label=pair_label)
        finally:
            plot_env.close()

        generated.extend(
            [
                path_file.name,
                reward_file.name,
                distance_file.name,
                action_file.name,
            ]
        )
        print_pair_summary(pair_spec.label, left_result, c_result)

    return generated


def evaluate_generated_pair_condition(
    args: argparse.Namespace,
    spec: GeneratedPairSpec,
) -> EvalResult:
    baseline_args = argparse.Namespace(
        baseline=spec.baseline,
        model_path=str(spec.model_path),
        out_dir=None,
        episodes=args.episodes,
        seed=args.seed,
        deterministic=args.deterministic,
        device=args.device,
        action_penalty_weight=args.action_penalty_weight,
        lowpass_alpha=args.lowpass_alpha,
        eval_lowpass=0,
    )
    result = evaluate_baseline(baseline_args)
    return EvalResult(label=spec.label, use_kf=spec.use_kf, episodes=result.episodes)


def print_pair_summary(left_label: str, left: EvalResult, c_result: EvalResult) -> None:
    left_summary = summarize(left)
    c_summary = summarize(c_result)
    print(
        f"{left_label} vs {c_result.label}: "
        f"return {left_summary['average_return']:.3f} vs {c_summary['average_return']:.3f}, "
        f"success {left_summary['success_rate']:.3f} vs {c_summary['success_rate']:.3f}, "
        f"exec_delta {left_summary['mean_executed_action_delta']:.6f} "
        f"vs {c_summary['mean_executed_action_delta']:.6f}"
    )


def write_readme(
    args: argparse.Namespace,
    out_dir: Path,
    copied: list[str],
    generated: list[str],
) -> None:
    copied_lines = "\n".join(f"- `{name}`" for name in copied)
    generated_lines = "\n".join(f"- `{name}`" for name in generated)
    content = f"""# C vs Other Complex Ablation Figures

This folder collects pairwise figures comparing `C_KF_in_loop` against the completed `group1_complex` conditions.

## Evaluation Settings

- Episodes for regenerated D/E/F/G comparisons: `{args.episodes}`
- Seeds: `{args.seed}` to `{args.seed + args.episodes - 1}`
- Deterministic: `{args.deterministic}`
- Device: `{args.device}`
- Low-pass alpha: `{args.lowpass_alpha}`
- Action penalty weight: `{args.action_penalty_weight}`

## Pair Definitions

- `A_vs_C`: A no-KF train/eval vs C KF-in-loop train/eval.
- `B_vs_C`: B no-KF train with external KF eval vs C KF-in-loop train/eval.
- `D_vs_C`: SAC with action-delta reward penalty vs C.
- `E_vs_C`: A policy evaluated with low-pass action filtering vs C.
- `F_vs_C`: SAC with gSDE vs C.
- `G_vs_C`: KF-in-loop without previous-executed-action observation augmentation vs C.

## Figure Types

- `*_path_complex.png`: selected best-return episode trajectories on the complex map.
- `*_reward_curve_complex.png`: selected episode step reward and cumulative reward.
- `*_step_distance_complex.png`: selected episode distance-to-goal.
- `*_action_smoothing_complex.png`: raw/executed action norms and action-delta norms.

## Copied From Existing AC/BC Outputs

{copied_lines}

## Regenerated By `python -m evaluate.export_c_path_compare_figs`

{generated_lines}
"""
    (out_dir / "README.md").write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
