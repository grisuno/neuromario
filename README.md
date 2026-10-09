# TopoMario

TopoExploit's TopoGPT2 architecture (quaternion spectral layers, 8-node torus
brain, sliding-window attention) repurposed as a policy network for Super Mario
Bros, plus the anti-stuck machinery ported from
[neuromario](https://github.com/grisuno/neuromario).

```bash
source /home/grisun0/LazyOwn/env/bin/activate

python3 topomario/train.py --timesteps 3000000 --stage 1-1
python3 play.py --episodes 5 --antistuck --no-viz      # headless
python3 play.py --episodes 5 --antistuck                # live dashboard
```

`DISPLAY` must be set for the dashboard (TkAgg). Pass `--no-viz` otherwise.

## Checkpoints: one slot, never lost

Both entry points share `TopoMarioAgent` (`topomario/agent.py`), so `train.py`
and `play.py` resume from and write to the **same** slot:

```
checkpoints_topomario/
├── last/
│   ├── model.safetensors     # weights (safetensors)
│   ├── optimizer.pt          # optimizer state, so training truly resumes
│   └── meta.json             # step + metrics
└── topomario_history.jsonl   # append-only metrics log
```

`CheckpointStore` guarantees:

- **Atomic** — writes to `.last.tmp`, then renames the directory into place.
- **Single slot** — `prune()` deletes every other checkpoint dir and every
  stray `.safetensors`/`.pt`, so no `step_*` junk accumulates.
- **Never lost** — saved after every PPO update and at the end of every episode,
  and `SIGINT`/`SIGTERM` (Ctrl-C) trigger a final save.
- **Resumable** — `agent.load()` restores weights *and* optimizer state, so
  `--timesteps` counts forward from where you stopped.

`play.py` learns while it plays: transitions from the episode are fed to PPO and
updated every `--rollout` steps, plus a flush at each episode end
(`--min-rollout`, default 64) so short episodes still produce a gradient step.
Use `--no-learn` to watch without updating.

## Anti-stuck tricks ported from NeuroMario

| Trick | Where | Value |
| --- | --- | --- |
| `FrameSkip` | `environment.FrameSkip` | 8 frames/action |
| `StuckMonitor` | `environment.StuckMonitor` | stuck 100, inactive 150, 2000 max |
| Curriculum goals | `environment.CurriculumGoal` | offset 30→120, adaptive ±15/±10 |
| Reward shaping | `environment.RewardShaper` | progress 0.1/px, goal +2.5, coin +1.2, score 0.015, death −2.0, stuck −3.0 |
| Exploration decay | `environment.ExplorationSchedule` | 0.985 default, floor 0.05 |
| Live dashboard | `play.Dashboard` | screen + saliency + action values |

The curriculum goal is the piece that actually unlocks progress: every episode
it re-targets `progress_x + offset`, raises the offset while the success rate is
high, and lowers it when the agent plateaus. Plus `--antistuck` in `play.py`
swaps idle/left actions for `RIGHT_A` once `stuck_steps > 5`.

## Bugs that made the agent play badly

Three of these made it pick `LEFT` forever at `confidence=0.215`:

1. **`_extract_semantic` read the wrong object.** It was handed the RGB array
   instead of the `info` dict, so all 27 semantic features were constant zeros —
   Mario's position never reached the policy. Fixed in `environment.py:_semantic`.
2. **Weight init crushed the logits.** `std=0.02` on every `nn.Linear` left the
   action decoder dominated by its bias, so argmax never moved. Now
   `kaiming_uniform_` (`model.py:_init_weights`).
3. **Double normalization.** The environment already returns `[0,1]`, but
   `inference.py`/`train.py` divided by 255 again.

Two more surfaced during training:

4. **The value function was `logits.max()`.** Using the greedy action logit as
   the critic distorts GAE and drove entropy to 0.004 (a fully deterministic
   policy). `ActionDecoder` now has a separate `value` head.
5. **`gym` fires `done=True` on death while `info['life']` still lags.** The
   wrapper now classifies termination itself into `died` / `level_complete` /
   `stuck` / `timeout`.

## Action space

`SIMPLE_MOVEMENT`: `NOOP, RIGHT, RIGHT_A, RIGHT_B, RIGHT_A_B, A, LEFT`
(`RIGHT_A` = run + jump, the workhorse).

## Layout

```
topomario/model.py         TopoMario: QuaternionLinear, QuaternionSpectralLayer,
                           SpectralAutoencoder, QuaternionTorusBrain, ActionDecoder
topomario/environment.py   env + FrameSkip/StuckMonitor/CurriculumGoal/RewardShaper
topomario/agent.py          TopoMarioAgent (PPO + atomic single-slot CheckpointStore)
topomario/train.py         training loop over the shared agent
topomario/inference.py     stateless action helper
play.py                    play + live dashboard
```

11.9M parameters at the `micro` preset.
