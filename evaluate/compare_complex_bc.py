"""Compare group-1 complex ablation B vs C."""

from __future__ import annotations

from evaluate.compare_complex_pair import parse_pair_args, run_pair_comparison


def main() -> None:
    args = parse_pair_args(
        description="Compare B external-KF baseline against C KF-in-loop on ComplexNavEnv.",
        default_out_dir="outputs/ablations/group1_complex/BC_compare",
    )
    run_pair_comparison("B", "C", args)


if __name__ == "__main__":
    main()
