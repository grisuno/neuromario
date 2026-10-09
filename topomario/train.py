#!/usr/bin/env python3
"""
Train TopoMario with PPO on the NeuroMario anti-stuck curriculum.

Shares ``TopoMarioAgent`` with ``play.py``, so both entry points resume from and
write to the single slot ``<root>/last``. Saving is atomic, prunes every other
checkpoint, and Ctrl-C still saves.
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--timesteps', type=int, default=3_000_000)
    p.add_argument('--rollout', type=int, default=512)
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--gae-lambda', type=float, default=0.95)
    p.add_argument('--clip', type=float, default=0.1)
    p.add_argument('--value-coef', type=float, default=0.5)
    p.add_argument('--entropy-coef', type=float, default=0.02)
    p.add_argument('--min-entropy', type=float, default=0.8)
    p.add_argument('--max-grad-norm', type=float, default=0.5)
    p.add_argument('--stage', default='1-1')
    p.add_argument('--device', default=None)
    p.add_argument('--epsilon-start', type=float, default=1.0)
    p.add_argument('--epsilon-floor', type=float, default=0.10)
    p.add_argument('--epsilon-decay', type=float, default=0.9999)
    p.add_argument('--max-steps', type=int, default=Config.MAX_AGENT_STEPS)
    p.add_argument('--envs', type=int, default=1,
                   help='parallel NES envs sharing one batched forward pass')
    p.add_argument('--no-diverse', dest='diverse', action='store_false',
                   default=True,
                   help='disable stratified action sampling across envs')
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--checkpoint-dir', default='checkpoints_topomario')
    p.add_argument('--log-every', type=int, default=5000)
    p.add_argument('--save-every', type=int, default=5000,
                   help='steps between checkpoints')
    p.add_argument('--from-scratch', action='store_true')
    p.add_argument('--antistuck', action='store_true', default=True)
    p.add_argument('--no-antistuck', dest='antistuck', action='store_false')
    p.add_argument('--cause-coef', type=float, default=0.5,
                   help='peso cabeza auxiliar PORQUE (BCE 16 causas)')
    p.add_argument('--micro-bs', type=int, default=0,
                   help='muestras por forward en CUDA (0=auto min(batch,8))')
    args = p.parse_args()

    store = CheckpointStore(args.checkpoint_dir)

    agent = TopoMarioAgent(device=args.device, store=store, lr=args.lr,
                           gamma=args.gamma, gae_lambda=args.gae_lambda,
                           clip=args.clip, value_coef=args.value_coef,
                           entropy_coef=args.entropy_coef,
                           min_entropy=args.min_entropy,
                           max_grad_norm=args.max_grad_norm,
                           epochs=args.epochs, batch_size=args.batch_size,
                           cause_coef=args.cause_coef,
                           micro_bs=args.micro_bs)

    resume_from = 0
    try:
        if store.exists() and not args.from_scratch:
            resume_from = agent.load()
            print(f'resumed from {store.slot} @ step {resume_from}')
        else:
            agent.acquire_lock()
            print('starting from scratch')
    except CheckpointLockError as exc:
        print(f'ERROR: {exc}')
        return 1

    env = (create_vector_env(n=args.envs, stage=args.stage,
                             max_steps=args.max_steps)
           if args.envs > 1 else
           create_environment(stage=args.stage, max_steps=args.max_steps))
    names = env.action_names
    n_envs = getattr(env, 'n', 1)
    vec = n_envs > 1

    recent = deque(maxlen=20)
    best_x = deque(maxlen=200)
    metrics = {}
    start_step = agent.step
    epsilon = args.epsilon_start
    last_ckpt = agent.step
    last_save = agent.step

    def current_metrics():
        return dict(metrics, reward20=float(np.mean(recent)) if recent else 0.0,
                    epsilon=epsilon, deaths=env.deaths,
                    levels_completed=env.levels_completed,
                    goal_offset=env.goal_offset if vec else env.shaper.goal.offset,
                    goal_success_rate=(env.goal_success_rate if vec else
                                       env.shaper.goal.success_rate))

    install_save_on_signal(agent, current_metrics)

    print(f'device={agent.device} envs={n_envs} actions={len(names)} '
          f'params={sum(p.numel() for p in agent.model.parameters()):,}')

    obs = env.reset()
    infos = [{} for _ in range(n_envs)]
    info = {}  # single-env path reads this directly
    ep_reward = np.zeros(n_envs)
    ep_max_x = np.zeros(n_envs, dtype=int)
    prev_cause = np.zeros(16, dtype=np.float32)
    prev_cause_b = np.zeros((n_envs, 16), dtype=np.float32)

    target = args.timesteps + resume_from
    while agent.step < target:
        if vec:
            force = None
            if args.antistuck:
                force = np.full(n_envs, -1, dtype=np.int64)
                for i in range(n_envs):
                    if infos[i].get('stuck_steps', 0) > 5:
                        force[i] = 2
            actions, logp, values, sources, _ = agent.act_batch(
                obs[0], obs[1], epsilon=epsilon, force=force,
                diverse=args.diverse, temperature=args.temperature,
                prev_cause=prev_cause_b)
            agent.observe_batch(obs[0], obs[1], actions, logp, values,
                                prev_cause=prev_cause_b)
            screens, sems, rewards_t, dones, infos = env.step(actions)
            obs = (screens, sems)
            agent.buffer.reward[-1] = rewards_t
            agent.buffer.done[-1] = dones.astype(np.float32)
            try:
                cv = np.stack([
                    np.asarray(inf.get('cause_vec',
                                       np.zeros(16, dtype=np.float32)),
                               dtype=np.float32) for inf in infos])
            except Exception:
                cv = np.zeros((n_envs, 16), dtype=np.float32)
            agent.buffer.cause[-1] = cv
            prev_cause_b = cv
            rewards_per_env = rewards_t
            info = infos[0]
        else:
            force = (2 if (args.antistuck and info.get('stuck_steps', 0) > 5)
                     else None)
            action, logp, value, source = agent.act(obs[0], obs[1],
                                                     epsilon=epsilon, force=force,
                                                     prev_cause=prev_cause)
            agent.observe(obs[0], obs[1], action, logp, 0.0, False, value,
                          prev_cause=prev_cause)
            obs0, obs1, reward, done, info = env.step(action)
            obs = (obs0, obs1)
            agent.buffer.reward[-1] = reward
            agent.buffer.done[-1] = float(done)
            cv = np.asarray(info.get('cause_vec',
                                     np.zeros(16, dtype=np.float32)),
                             dtype=np.float32)
            agent.buffer.cause[-1] = cv
            prev_cause = cv
            rewards_per_env = np.array([reward], dtype=np.float32)
            dones = np.array([done])

        ep_reward += rewards_per_env
        for i in range(n_envs):
            ep_max_x[i] = max(ep_max_x[i], infos[i].get('progress_x', 0))
        epsilon = max(args.epsilon_floor, epsilon * args.epsilon_decay)

        if dones.any():
            recent.append(float(ep_reward.mean()))
            best_x.append(int(ep_max_x.max()))
            obs = env.reset()
            infos = [{} for _ in range(n_envs)]
            info = {}
            ep_reward = np.zeros(n_envs)
            ep_max_x = np.zeros(n_envs, dtype=int)

        if not agent.ready(args.rollout):
            continue

        out = agent.learn()
        if not out:
            continue
        metrics = dict(out)
        metrics.update(current_metrics())
        metrics['ep_max_x'] = float(np.mean(best_x)) if best_x else 0.0

        if agent.step - last_save >= args.log_every:
            last_save = agent.step
            mix = metrics['action_mix']
            print(f'step {agent.step:8d} | reward {metrics["reward20"]:8.1f} | '
                  f'max_x {metrics["ep_max_x"]:5.0f} | '
                  f'pl {metrics["policy_loss"]:+.4f} | '
                  f'vl {metrics["value_loss"]:8.2f} | '
                  f'H {metrics["entropy"]:.4f} | '
                  f'kl {metrics["approx_kl"]:.5f} | '
                  f'causa L={metrics.get("cause_loss", 0.0):.4f} '
                  f'acc={metrics.get("cause_acc", 0.0):.2f} | '
                  f'eps {epsilon:.3f} | goal+{metrics["goal_offset"]} '
                  f'({metrics["goal_success_rate"]:.2f}) | '
                  f'deaths {env.deaths} | mix {mix}')

        if agent.step - last_ckpt >= args.save_every:
            last_ckpt = agent.step
            agent.save(metrics)
            print(f'  [saved] {store.slot} @ {agent.step}')

    env.close()
    agent.save({**current_metrics(), 'final': True})
    print(f'done. checkpoint: {store.slot} @ step {agent.step}')


if __name__ == '__main__':
    sys.exit(main() or 0)
