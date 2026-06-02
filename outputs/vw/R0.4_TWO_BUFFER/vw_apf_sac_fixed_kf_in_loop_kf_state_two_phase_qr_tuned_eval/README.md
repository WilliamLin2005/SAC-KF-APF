# Robot v/w Fixed-KF Two-Phase Eval

## Setup

- Model path: `/home/ubt/path_planning/SAC-KF-APF/outputs/vw/ablation/vw_apf_sac_fixed_kf_in_loop_kf_state_two_phase_qr_tuned.zip`
- smoother: `fixed`
- obs_mode: `kf_state`
- slowdown radius: `8.0`
- success thresholds: `v_exec <= 0.25`, `|w_exec| <= 0.25`
- Episodes: `50`
- Seeds: `0` to `49`
- Deterministic: `0`

## Outputs

- `path_complex.png`: selected best-return trajectory.
- `eval_reward_curve_complex.png`: selected episode reward and cumulative reward.
- `eval_step_distance_complex.png`: selected episode distance-to-goal.
- `command_smoothing_complex.png`: raw/executed `[v,w]`, command deltas, and filter mismatch.
- `episode_metrics.csv`: per-episode two-phase and terminal command metrics.
- `metrics_summary.csv`: aggregate metrics.

## Key Summary

- Success rate: `1.000`
- Average return: `531.299`
- Average steps: `108.740`
- Mean docking fraction: `0.087942`
- Mean terminal v_exec: `0.209388`
- Mean terminal |w_exec|: `0.038004`
- Mean last-10 v_exec: `0.592345`
- Mean last-10 |w_exec|: `0.067647`
