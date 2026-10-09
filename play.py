#!/usr/bin/env python3
"""
Play Super Mario Bros with TopoMario.

Learns online while it plays (PPO updates from the transitions it generates) and
checkpoints into the single slot ``<root>/last``. Saving is atomic and prunes
every other checkpoint, and Ctrl-C still saves.
"""
from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from topomario.agent import (CheckpointLockError, CheckpointStore,
                             TopoMarioAgent, install_save_on_signal)
from topomario.environment import (Config, create_environment,
                                    create_vector_env)


class Dashboard:
    """Live view of every parallel env plus the action values.

    With ``--envs N`` all N NES instances are shown side by side, so the vectorized
    rollouts are visible instead of being reduced to one screen. Single-env runs
    keep the original 3-panel layout (screen / focus / action values).
    """

    def __init__(self, action_names, enabled=True, n_envs=1):
        self.action_names = list(action_names)
        self.n_envs = max(1, n_envs)
        self.enabled = enabled
        self.fig = None
        self.axes = {}
        if not enabled:
            return
        try:
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
            self.plt = plt
            n = self.n_envs
            if n == 1:
                self.fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                self.axes = {'screens': [axes[0]], 'focus': [axes[1]],
                             'values': axes[2]}
                axes[0].set_title('Game screen (model input)')
                axes[1].set_title('Focus')
                axes[2].set_title('Action values')
            else:
                cols = min(4, n)
                rows = (n + cols - 1) // cols
                self.fig, all_axes = plt.subplots(rows * 2 + 1, cols,
                                                   figsize=(4.2 * cols, 3.1 * rows * 2))
                all_axes = all_axes.reshape(-1)
                screens, focus = [], []
                for i in range(n):
                    top = all_axes[i]
                    bot = all_axes[rows * cols + i]
                    top.set_title(f'env {i}  (input)', fontsize=9)
                    bot.set_title(f'env {i}  (focus)', fontsize=8)
                    top.axis('off')
                    bot.axis('off')
                    screens.append(top)
                    focus.append(bot)
                self.axes = {'screens': screens, 'focus': focus,
                             'values': all_axes[rows * 2 * cols]}
                for ax in all_axes[n:rows * 2 * cols]:
                    ax.axis('off')
            plt.ion()
            self.fig.canvas.manager.set_window_title(
                f'TopoMario live ({self.n_envs} envs)')
        except Exception as exc:
            print(f'[dashboard] disabled ({exc})')
            self.enabled = False

    def update(self, screens, focus, values, infos, chosen, eps):
        """``screens`` is [N,12,H,W] con frame-stack (o [N,3,H,W] clasico)."""
        if not self.enabled:
            return
        plt = self.plt

        def _rgb(s):
            return s[:3] if s.shape[0] > 3 else s

        if self.n_envs == 1:
            img = np.transpose(_rgb(screens), (1, 2, 0))
            ax = self.axes['screens'][0]
            ax.clear()
            ax.imshow(img)
            ax.axis('off')
            ax = self.axes['focus']
            ax.clear()
            ax.imshow(img)
            ax.imshow(focus, cmap='inferno', alpha=0.55)
            ax.axis('off')
            vals_axis = self.axes['values']
            colors = ['red' if i == chosen else 'steelblue'
                      for i in range(len(self.action_names))]
            vals_axis.clear()
            vals_axis.barh(self.action_names, values, color=colors)
            vals_axis.set_title(f'eps={eps:.2f}', fontsize=9)
            vals_axis.axvline(0.0, color='k', lw=0.8)
        else:
            for i in range(self.n_envs):
                img = np.transpose(_rgb(screens[i]), (1, 2, 0))
                info = infos[i] if infos is not None else {}
                ax = self.axes['screens'][i]
                ax.clear()
                ax.imshow(img)
                ax.set_title(f"env{i}  x={info.get('x_pos', 0)}  "
                             f"stuck={info.get('stuck_steps', 0)}", fontsize=8)
                ax.axis('off')
                ax = self.axes['focus'][i]
                ax.clear()
                ax.imshow(img)
                ax.imshow(focus[i], cmap='inferno', alpha=0.55)
                ax.axis('off')
            ax = self.axes['values']
            ax.clear()
            counts = np.bincount(np.asarray(chosen, dtype=int),
                                 minlength=len(self.action_names))
            colors = ['tomato' if c > 0 else 'steelblue'
                      for c in counts]
            ax.barh(self.action_names, counts, color=colors)
            ax.set_title(f'action counts  eps={eps:.2f}', fontsize=9)

        self.fig.suptitle(self._title(infos), fontsize=9)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    @staticmethod
    def _title(infos):
        if not infos:
            return ''
        parts = []
        for i, info in enumerate(infos):
            parts.append(f"{i}:x={info.get('x_pos', 0)}/{info.get('goal_target', 0)}"
                         f"{'*' if info.get('died') else ''}"
                         f"{'!' if info.get('stuck') else ''}")
        return '  '.join(parts)

    def close(self):
        if self.enabled and self.fig is not None:
            self.plt.close(self.fig)

def summarize_end(infos):
    """Count the end condition across every env, not just infos[0].

    Only looking at env 0 made any round whose env 0 happened to have no flag
    print as ``env_done``, which reads like "level finished" when it just means
    "we could not tell".
    """
    if not infos:
        return ['unknown']
    counts = {}
    for info in infos:
        for key in ('level_complete', 'died', 'stuck', 'inactive', 'timeout'):
            if info.get(key):
                counts[key] = counts.get(key, 0) + 1
    if not counts:
        return [f'undetermined x{len(infos)}']
    parts = [f'{k}x{v}' for k, v in sorted(counts.items(), key=lambda kv: -kv[1])]
    if counts.get('level_complete'):
        parts.append(f'FLAG/1-{counts["level_complete"]}')
    return parts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint-dir', default='checkpoints_topomario')
    p.add_argument('--episodes', type=int, default=5,
                   help='rounds; each round steps all --envs in parallel')
    p.add_argument('--envs', type=int, default=4,
                   help='parallel NES envs sharing one batched forward pass')
    p.add_argument('--frameskip', type=int, default=Config.FRAMESKIP,
                   choices=range(1, 11), metavar='1-10',
                   help='NES frames per action. 1 = no skip (normal speed), '
                        '10 = fastest. Each decision advances the game '
                        'frameskip frames, so Mario moves 10x faster at 10.')
    p.add_argument('--no-diverse', dest='diverse', action='store_false',
                   default=True,
                   help='disable stratified action sampling across envs')
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--max-steps', type=int, default=Config.MAX_AGENT_STEPS)
    p.add_argument('--stage', default='1-1')
    p.add_argument('--device', default=None)
    p.add_argument('--epsilon', type=float, default=0.15,
                   help='exploration rate while playing')
    p.add_argument('--no-learn', action='store_true',
                   help='watch only, do not run PPO updates')
    p.add_argument('--cause-coef', type=float, default=0.5,
                   help='peso cabeza auxiliar PORQUE (BCE 16 causas)')
    p.add_argument('--rollout', type=int, default=512,
                   help='steps between PPO updates while playing')
    p.add_argument('--min-rollout', type=int, default=512,
                   help='minimum banked transitions (envs x ticks) per update')
    p.add_argument('--save-every', type=int, default=1,
                   help='save after this many PPO updates (0 = only at exit)')
    p.add_argument('--antistuck', action='store_true', default=True)
    p.add_argument('--no-antistuck', dest='antistuck', action='store_false')
    p.add_argument('--no-viz', action='store_true')
    p.add_argument('--from-scratch', action='store_true',
                   help='ignore any existing checkpoint')
    args = p.parse_args()

    store = CheckpointStore(args.checkpoint_dir)
    agent = TopoMarioAgent(device=args.device, store=store,
                           cause_coef=args.cause_coef)

    try:
        if store.exists() and not args.from_scratch:
            step = agent.load()
            print(f'resumed from {store.slot} @ step {step}')
        else:
            agent.acquire_lock()
            print('starting from scratch')
    except CheckpointLockError as exc:
        print(f'ERROR: {exc}')
        return 1

    env = (create_vector_env(n=args.envs, stage=args.stage,
                             frame_skip=args.frameskip,
                             max_steps=args.max_steps)
           if args.envs > 1 else
           create_environment(stage=args.stage, frame_skip=args.frameskip,
                              max_steps=args.max_steps))
    names = env.action_names
    n_envs = getattr(env, 'n', 1)
    dash = Dashboard(names, enabled=not args.no_viz, n_envs=n_envs)

    recent = deque(maxlen=20)
    metrics = {}
    updates_since_save = [0]
    ep_mixes: list = []

    def current_metrics():
        return dict(metrics, episodes_done=len(recent),
                    deaths=env.deaths, levels_completed=env.levels_completed)

    install_save_on_signal(agent, current_metrics)

    rewards, progresses, mixes = [], [], []
    px_per_kf, reward_per_kf = [], []
    banked = 0

    def flush():
        """Run a PPO update once enough transitions are banked.

        With ``--envs 4`` each stepped tick contributes 4 transitions, so the
        floor is reached in a quarter of the wall-clock steps. Short episodes
        keep feeding the same buffer, which accumulates across rounds.
        """
        if args.no_learn:
            return
        if agent.banked() < args.min_rollout:
            return
        out = agent.learn()
        if not out:
            return
        metrics.clear()
        metrics.update(out)
        metrics.update(current_metrics())
        mixes.append(out['action_mix'])
        updates_since_save[0] += 1
        print(f'  upd#{updates_since_save[0]} step {agent.step} | '
              f'pl {out["policy_loss"]:+.4f} vl {out["value_loss"]:8.3f} '
              f'H {out["entropy"]:.4f} kl {out["approx_kl"]:.5f} | '
              f'causa L={out.get("cause_loss", 0.0):.4f} '
              f'acc={out.get("cause_acc", 0.0):.2f} | '
              f'mix {out["action_mix"]}')
        if args.save_every and updates_since_save[0] >= args.save_every:
            agent.save(metrics)
            updates_since_save[0] = 0

    for ep in range(1, args.episodes + 1):
        screens, semantics = env.reset()
        infos = [{} for _ in range(n_envs)]
        ep_reward = np.zeros(n_envs)
        steps = 0
        max_x = np.zeros(n_envs, dtype=int)
        counts = np.zeros(len(names), dtype=int)
        alive = np.ones(n_envs, dtype=bool)
        prev_cause = np.zeros((n_envs, 16), dtype=np.float32)

        while steps < args.max_steps and alive.any():
            force = None
            if args.antistuck:
                force = np.full(n_envs, -1, dtype=np.int64)
                for i in range(n_envs):
                    if infos[i].get('stuck_steps', 0) > 5:
                        force[i] = 2

            actions, logp, values, sources, logits = agent.act_batch(
                screens, semantics, epsilon=args.epsilon, force=force,
                diverse=args.diverse, temperature=args.temperature,
                prev_cause=prev_cause)

            agent.observe_batch(screens, semantics, actions, logp, values,
                                prev_cause=prev_cause)
            screens, semantics, rewards_t, dones, infos = env.step(actions)
            agent.buffer.reward[-1] = rewards_t
            agent.buffer.done[-1] = dones.astype(np.float32)
            # PORQUE exacto por env: vector causa del paso que acaba de pasar
            try:
                cv = np.stack([
                    np.asarray(inf.get('cause_vec',
                                       np.zeros(16, dtype=np.float32)),
                               dtype=np.float32) for inf in infos])
            except Exception:
                cv = np.zeros((n_envs, 16), dtype=np.float32)
            agent.buffer.cause[-1] = cv
            prev_cause = cv
            banked += n_envs

            steps += 1
            ep_reward += rewards_t
            for i in range(n_envs):
                if alive[i]:
                    max_x[i] = max(max_x[i], infos[i].get('progress_x', 0))
                    counts[actions[i]] += 1
                    if dones[i]:
                        alive[i] = False

            if not args.no_viz:
                h, w = screens.shape[2], screens.shape[3]
                focus = np.zeros((n_envs, h, w), dtype=np.float32)
                for i in range(n_envs):
                    col = int(np.clip(infos[i].get('x_pos', 0), 0, w - 1))
                    focus[i, :, col] = 1.0
                if n_envs == 1:
                    dash.update(screens[0], focus[0], logits[0].cpu().numpy(),
                                infos, int(actions[0]), args.epsilon)
                else:
                    dash.update(screens, focus, None, infos, actions, args.epsilon)

            flush()

            if steps % 200 == 0:
                print(f'  round{ep} tick{steps:5d} r={ep_reward.mean():8.1f} '
                      f'x={int(max_x.max()):5d}/{infos[0].get("goal_target", 0):5d} '
                      f'banked={banked} goal+{env.goal_offset} '
                      f'stuck={infos[0].get("stuck_steps", 0):3d}')

        reasons = summarize_end(infos)
        recent.append(float(ep_reward.mean()))
        rewards.append(float(ep_reward.mean()))
        progresses.append(int(max_x.max()))
        ep_mixes.append((counts / max(1, counts.sum())).round(3).tolist())
        frames = steps * args.frameskip
        px_per_kf.append(1000.0 * int(max_x.max()) / max(frames, 1))
        reward_per_kf.append(1000.0 * float(ep_reward.mean()) / max(frames, 1))
        print(f'round{ep}: reward={ep_reward.mean():.1f} ticks={steps} '
              f'frames={frames} max_x={int(max_x.max())} '
              f'px/1k_frames={1000.0 * int(max_x.max()) / max(frames, 1):.0f} '
              f'end={reasons} '
              f'dist={np.round(counts / max(1, counts.sum()), 2)} '
              f'banked={banked}')
        flush()
        agent.save({**current_metrics(), 'episode_reward': float(ep_reward.mean()),
                    'max_x': int(max_x.max()), 'frames': frames,
                    'end': ','.join(reasons)})

    env.close()
    dash.close()
    agent.save({**current_metrics(), 'final': True})

    mix = np.zeros(len(names))
    source_mixes = mixes if mixes else ep_mixes
    if source_mixes:
        mix = np.array(source_mixes).mean(axis=0)
    print(f'ppo updates this run: {len(mixes)}'
          f'{" (watch-only)" if args.no_learn else ""}'
          f'  transitions banked: {banked}')
    print('=' * 60)
    print(f'envs={n_envs}  frameskip={args.frameskip}  rounds={args.episodes}')
    print(f'mean_reward={np.mean(rewards):.1f} +/- {np.std(rewards):.1f}  '
          f'reward/1k_frames={np.mean(reward_per_kf):.1f}')
    print(f'mean_max_x={np.mean(progresses):.1f}  best={max(progresses)}  '
          f'mean_px/1k_frames={np.mean(px_per_kf):.0f}')
    print(f'checkpoint: {store.slot} @ step {agent.step}')
    print('learned action mix:', {names[i]: float(mix[i]) for i in range(len(names))})
    print(f'idle+reverse share={mix[0] + mix[-1]:.1%}')
    print(f'deaths={env.deaths}  levels_completed={env.levels_completed}  '
          f'flag_get_fired={env.levels_completed > 0}')
    if env.levels_completed == 0:
        print('  NOTE: no flag_get -> the flagpole was never touched, so no '
              'round completed the level regardless of max_x.')
    print('=' * 60)


if __name__ == '__main__':
    sys.exit(main() or 0)
