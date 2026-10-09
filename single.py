#!/usr/bin/env python3
"""
single.py — una sola instancia visible con entrenamiento continuo resumible.

Diferencias con play.py:
- siempre 1 env (ves lo que ve la red, sin vectorizar).
- entrenamiento infinito por defecto (--episodes 0 = sin fin, Ctrl-C guarda).
- --model-size elige la red: micro/small/medium/large/gpt2/xl/xxl.
  xl/xxl son mas grandes (768-1024 dim, 16 capas, 4 spectral layers, MoE-8)
  porque small no aprendia ni en una noche.
- foco real de la red (saliency MarioVision) en vez de columna falsa.
- resume automatico del slot <root>/last salvo --from-scratch.

Flags: tamano modelo, frameskip y todo lo necesario para aprender (lr, clip,
entropy, epsilon schedule, rollout, ae-weight, stage, device...).
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import deque
from pathlib import Path

# Debe fijarse ANTES de importar torch: permite que el allocator CUDA crezca
# por segmentos en vez de reservar bloques fijos, lo que evita el OOM por
# fragmentacion ("reserved but unallocated") del que avisa el propio error.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from topomario.agent import (CheckpointLockError, CheckpointStore,
                             TopoMarioAgent, install_save_on_signal)
from topomario.environment import Config, create_environment

MODEL_SIZES = ['micro', 'small', 'medium', 'large', 'gpt2', 'xl', 'xxl']


class MetricsTracker:
    """Curvas acumulativas estilo neuromario (scores/losses/accuracies).

    Todo update y episodio se guarda en memoria y en CSV acumulativo, asi
    la proxima vez que entrenas ves el historial completo y el modelo sigue
    donde quedo (entrenamiento acumulativo real).
    """

    POS_KEYS = ['progress', 'goal', 'coin', 'score', 'vertical',
                'enemy', 'block', 'powerup', 'secret']
    NEG_KEYS = ['idle', 'backtrack', 'damage', 'death', 'samespot',
                'stuck', 'inactive']

    def __init__(self, csv_path: Path):
        self.csv_path = Path(csv_path)
        self.ep_reward: list = []
        self.ep_max_x: list = []
        self.ep_acc: list = []      # goal_success_rate por episodio
        self.ep_eps: list = []
        self.ep_pos: list = []      # suma de rewards positivos del episodio
        self.ep_neg: list = []      # suma de castigos (negativo) del episodio
        self.pl: list = []          # policy loss por update
        self.vl: list = []          # value loss por update
        self.ent: list = []
        self.kl: list = []
        self._load_csv()

    @staticmethod
    def _f(row: dict, key: str, default: float = 0.0) -> float:
        try:
            v = row.get(key, default)
            if v in (None, ''):
                return default
            return float(v)
        except (TypeError, ValueError):
            return default

    def _load_csv(self):
        try:
            if self.csv_path.exists():
                import csv
                with open(self.csv_path) as fh:
                    for row in csv.DictReader(fh):
                        try:
                            kind = row.get('kind', 'ep')
                            if kind == 'ep':
                                self.ep_reward.append(self._f(row, 'reward'))
                                self.ep_max_x.append(self._f(row, 'max_x'))
                                self.ep_acc.append(self._f(row, 'acc'))
                                self.ep_eps.append(self._f(row, 'eps'))
                                self.ep_pos.append(self._f(row, 'pos'))
                                self.ep_neg.append(self._f(row, 'neg'))
                            else:
                                self.pl.append(self._f(row, 'pl'))
                                self.vl.append(self._f(row, 'vl'))
                                self.ent.append(self._f(row, 'ent'))
                                self.kl.append(self._f(row, 'kl'))
                        except (ValueError, KeyError, TypeError):
                            continue
                print(f'[metrics] historial cargado: {len(self.ep_reward)} eps, '
                      f'{len(self.pl)} updates ({self.csv_path})')
        except OSError:
            pass

    def _append_csv(self, row: dict):
        try:
            new = not self.csv_path.exists()
            import csv
            with open(self.csv_path, 'a', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=list(row.keys()))
                if new:
                    w.writeheader()
                w.writerow(row)
        except OSError:
            pass

    def log_episode(self, reward: float, max_x: int, acc: float, eps: float,
                    pos: float = 0.0, neg: float = 0.0):
        self.ep_reward.append(float(reward))
        self.ep_max_x.append(float(max_x))
        self.ep_acc.append(float(acc))
        self.ep_eps.append(float(eps))
        self.ep_pos.append(float(pos))
        self.ep_neg.append(float(neg))
        self._append_csv({'kind': 'ep', 'reward': reward, 'max_x': max_x,
                          'acc': acc, 'eps': eps, 'pos': pos, 'neg': neg,
                          'pl': '', 'vl': '', 'ent': '', 'kl': ''})

    def log_update(self, pl: float, vl: float, ent: float, kl: float):
        self.pl.append(float(pl))
        self.vl.append(float(vl))
        self.ent.append(float(ent))
        self.kl.append(float(kl))
        self._append_csv({'kind': 'upd', 'reward': '', 'max_x': '', 'acc': '',
                          'eps': '', 'pos': '', 'neg': '',
                          'pl': pl, 'vl': vl, 'ent': ent, 'kl': kl})

    @staticmethod
    def _clean(xs, fill=0.0):
        import math
        return [x if isinstance(x, (int, float)) and math.isfinite(x) else fill
                for x in xs]


def get_focus(agent: TopoMarioAgent, screen: np.ndarray) -> np.ndarray:
    """Mapa de atencion real [H,W] en [0,1]. Fallback: columna en x_pos."""
    try:
        agent.model.eval()
        with torch.no_grad():
            s = torch.from_numpy(screen).float().unsqueeze(0).to(agent.device)
            f = agent.model.state_encoder.focus_map(s)  # [1,1,fh,fw]
            f = f[0, 0].cpu().numpy()
        # upsample nearest a 240x256 aprox (pantalla normalizada [3,H,W])
        h, w = screen.shape[1], screen.shape[2]
        fh, fw = f.shape
        f = np.repeat(np.repeat(f, max(1, h // fh), axis=0), max(1, w // fw), axis=1)
        return f[:h, :w].astype(np.float32)
    except Exception:
        h, w = screen.shape[1], screen.shape[2]
        return np.zeros((h, w), dtype=np.float32)


class Dashboard:
    """Ventana 1 (juego): pantalla / foco real / valores. Ventana 2 (metricas
    estilo neuromario 2x2 dark): reward+max_x | pl+vl+ent+kl | acc+eps | mix."""

    def __init__(self, action_names, enabled=True, tracker: MetricsTracker | None = None):
        self.action_names = list(action_names)
        self.enabled = enabled
        self.fig = None
        self.mfig = None
        self.tracker = tracker
        self._ticks = 0
        # cache del panel [1,1]: update() redibuja cada 20 ticks SIN args y
        # eso borraba el panel hasta el fin del episodio. Ahora se reusa lo
        # ultimo visto hasta que llegue dato nuevo.
        self._last_mix = None
        self._last_breakdown = None
        self._last_extra = ''
        if not enabled:
            return
        try:
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
            self.plt = plt
            self.fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            self.ax_screen, self.ax_focus, self.ax_vals = axes
            self.ax_screen.set_title('Game screen (model input)')
            self.ax_focus.set_title('Focus (red: mira la red)')
            self.ax_vals.set_title('Action values')
            plt.ion()
            self.fig.canvas.manager.set_window_title('single.py juego (1 env)')
            # --- ventana metricas ---
            self.mfig, self.maxs = plt.subplots(2, 2, figsize=(14, 9))
            self.mfig.patch.set_facecolor('#0a0a0a')
            self.mfig.suptitle('single.py metricas PPO (acumulativo)',
                               fontsize=14, color='#00ff00', weight='bold')
            self.mfig.tight_layout(pad=4.0)
            for ax in self.maxs.flat:
                ax.set_facecolor('#111')
                ax.grid(True, color='#333', alpha=0.3)
                ax.tick_params(colors='white')
            self.mfig.canvas.manager.set_window_title('single.py metricas')
        except Exception as exc:
            print(f'[dashboard] disabled ({exc})')
            self.enabled = False

    def update(self, screen, focus, values, info, chosen, eps):
        if not self.enabled:
            return
        self._ticks += 1
        # frame stack [12,H,W] -> muestra el frame mas reciente (3 primeros
        # canales) como RGB; si es [3,H,W] clasico va directo
        rgb = screen[:3] if screen.shape[0] > 3 else screen
        img = np.transpose(rgb, (1, 2, 0))
        self.ax_screen.clear()
        self.ax_screen.imshow(img)
        self.ax_screen.axis('off')
        self.ax_screen.set_title(
            f"x={info.get('x_pos', 0)}/{info.get('goal_target', 0)} "
            f"stuck={info.get('stuck_steps', 0)}", fontsize=9)
        self.ax_focus.clear()
        self.ax_focus.imshow(img)
        self.ax_focus.imshow(focus, cmap='inferno', alpha=0.55)
        self.ax_focus.axis('off')
        colors = ['red' if i == chosen else 'steelblue'
                  for i in range(len(self.action_names))]
        self.ax_vals.clear()
        self.ax_vals.barh(self.action_names, values, color=colors)
        self.ax_vals.set_title(f'eps={eps:.3f}', fontsize=9)
        self.ax_vals.axvline(0.0, color='k', lw=0.8)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self.plt.pause(0.001)
        # metricas cada 20 ticks para no fundir la UI (igual que neuromario)
        if self._ticks % 20 == 0:
            self.update_metrics()

    def update_metrics(self, mix=None, breakdown=None, extra_text=''):
        """Redibuja la ventana 2x2. Sin tracker no hace nada.

        [1,1] = anatomia del reward del ultimo episodio: verde suma,
        rojo resta (idle/backtrack/damage/death/samespot/stuck). El mix de
        acciones va como texto para no perderlo.
        Los redibujos periodicos vienen sin args: usan el cache, no borran.
        """
        if not self.enabled or self.tracker is None:
            return
        if mix is not None:
            self._last_mix = mix
        if breakdown is not None:
            self._last_breakdown = breakdown
        if extra_text:
            self._last_extra = extra_text
        mix = self._last_mix
        breakdown = self._last_breakdown
        extra_text = self._last_extra
        t = self.tracker
        axs = self.maxs
        for ax in axs.flat:
            ax.clear()
            ax.set_facecolor('#111')
            ax.grid(True, color='#333', alpha=0.3)
            ax.tick_params(colors='white')
        W = 100  # ventana movil como neuromario
        # [0,0] reward + max_x por episodio
        if t.ep_reward:
            r = MetricsTracker._clean(t.ep_reward[-W:])
            axs[0, 0].plot(r, color='#00ff00', lw=2, marker='o', markersize=2,
                            label=f'reward (last {r[-1]:.1f})')
            axs[0, 0].set_title('Reward por episodio', color='#00ffff',
                                fontsize=11, weight='bold')
            axs[0, 0].legend(facecolor='black', labelcolor='white', fontsize=8)
            if t.ep_max_x:
                ax2 = axs[0, 0].twinx()
                ax2.plot(MetricsTracker._clean(t.ep_max_x[-W:]), color='#ffaa00',
                         lw=1.5, ls='--', label='max_x')
                ax2.tick_params(colors='#ffaa00')
        # [0,1] pl + vl + ent + kl por update
        if t.pl:
            n = min(W, len(t.pl))
            x = range(len(t.pl) - n, len(t.pl))
            axs[0, 1].plot(x, MetricsTracker._clean(t.pl[-n:]), color='#ff3333',
                            lw=2, label='policy_loss')
            axs[0, 1].plot(x, MetricsTracker._clean(t.vl[-n:], fill=0.0),
                            color='#ffaa00', lw=1.5, ls='--', label='value_loss')
            axs[0, 1].plot(x, MetricsTracker._clean(t.ent[-n:]), color='#00ffff',
                            lw=1.5, ls=':', label='entropy')
            axs[0, 1].set_title('Loss / entropy / KL por update', color='#00ff00',
                                fontsize=11, weight='bold')
            axs[0, 1].legend(facecolor='black', labelcolor='white', fontsize=8)
        # [1,0] acc metas + epsilon
        if t.ep_acc:
            axs[1, 0].plot(MetricsTracker._clean(t.ep_acc[-W:]), color='#99ccff',
                            lw=2.5, marker='s', markersize=2, label='acc metas')
            axs[1, 0].plot(MetricsTracker._clean(t.ep_eps[-W:]), color='#ff66cc',
                            lw=1.5, ls='--', label='epsilon')
            axs[1, 0].set_ylim(-0.05, 1.05)
            axs[1, 0].set_title('Acc (goal success) + epsilon', color='#ff00ff',
                                fontsize=11, weight='bold')
            axs[1, 0].legend(facecolor='black', labelcolor='white', fontsize=8)
        # [1,1] anatomia reward: +verde / -rojo del ultimo episodio
        bd = breakdown or {}
        if bd:
            keys = [k for k in MetricsTracker.POS_KEYS + MetricsTracker.NEG_KEYS
                    if abs(bd.get(k, 0.0)) > 1e-9]
            if keys:
                vals = [bd[k] for k in keys]
                cols = ['#00ff88' if v >= 0 else '#ff3344' for v in vals]
                axs[1, 1].barh(keys, vals, color=cols, edgecolor='white')
                axs[1, 1].axvline(0.0, color='white', lw=0.8)
                axs[1, 1].set_title('+suma / -resta (ultimo ep)', color='#ffaa00',
                                    fontsize=11, weight='bold')
                for k, v in zip(keys, vals):
                    axs[1, 1].text(v, k, f' {v:+.1f}', va='center',
                                   color='white', fontsize=8, weight='bold')
        elif mix is not None:
            y = np.asarray(mix, dtype=float)
            cols = ['#ff0000' if v == y.max() and v > 0 else '#00ffff'
                    for v in y]
            axs[1, 1].barh(self.action_names, y, color=cols, edgecolor='white')
            axs[1, 1].set_title('Action mix (media)', color='#ffaa00',
                                fontsize=11, weight='bold')
        if extra_text:
            axs[1, 1].text(0.98, 0.02, extra_text, transform=axs[1, 1].transAxes,
                           color='white', fontsize=8, ha='right', va='bottom',
                           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        self.mfig.canvas.draw_idle()
        self.mfig.canvas.flush_events()
        self.plt.pause(0.001)

    def close(self):
        if self.enabled:
            try:
                if self.fig is not None:
                    self.plt.close(self.fig)
                if self.mfig is not None:
                    self.plt.close(self.mfig)
            except Exception:
                pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    # --- modelo ---
    p.add_argument('--model-size', default='medium', choices=MODEL_SIZES,
                   help='tamano red: micro/small/medium/large/gpt2/xl/xxl (default medium)')
    p.add_argument('--ae-weight', type=float, default=0.01,
                   help='peso recon espectral+visual (AE_RECON_WEIGHT)')
    # --- env ---
    p.add_argument('--frameskip', type=int, default=Config.FRAMESKIP,
                   choices=range(1, 11), metavar='1-10',
                   help='NES frames por accion (1 lento, 10 rapido)')
    p.add_argument('--stage', default='1-1')
    p.add_argument('--max-steps', type=int, default=Config.MAX_AGENT_STEPS)
    p.add_argument('--stuck-limit', type=int, default=Config.STUCK_LIMIT)
    p.add_argument('--inactivity-limit', type=int, default=Config.INACTIVITY_LIMIT)
    # --- run ---
    p.add_argument('--episodes', type=int, default=0,
                   help='episodios; 0 = infinito hasta Ctrl-C (default 0)')
    p.add_argument('--no-learn', action='store_true', help='solo mirar, sin PPO')
    p.add_argument('--rollout', type=int, default=512, help='ticks entre updates PPO')
    p.add_argument('--save-every', type=int, default=1,
                   help='guarda cada N updates (0 = solo al salir/morir)')
    p.add_argument('--no-viz', action='store_true')
    # --- PPO ---
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--gae-lambda', type=float, default=0.95)
    p.add_argument('--clip', type=float, default=0.1)
    p.add_argument('--value-coef', type=float, default=0.5)
    p.add_argument('--entropy-coef', type=float, default=0.02)
    p.add_argument('--cause-coef', type=float, default=0.5,
                   help='peso cabeza auxiliar PORQUE (BCE 16 causas). '
                        '0 = solo escalar opaco, como antes')
    p.add_argument('--min-entropy', type=float, default=0.8)
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=None,
                   help='minibatch PPO (default auto por tamano: '
                        'micro/small/medium 64, large/gpt2 32, xl 16, xxl 4)')
    p.add_argument('--micro-bs', type=int, default=0,
                   help='muestras por forward en CUDA (0=auto min(batch,8)). '
                        'xxl en 6GB: pon 1 o 2')
    p.add_argument('--optimizer', default=None, choices=['adam', 'sgd', 'adafactor'],
                   help='adam=2 estados/param (xxl OOM en 6GB), '
                        'sgd=1 estado, adafactor=factorizado. '
                        'default auto: adam salvo xl/xxl que usan adafactor')
    p.add_argument('--amp', dest='amp', action='store_true', default=None,
                   help='autocast fp16 en el update: -40%% VRAM aprox '
                        '(auto-on en xl/xxl)')
    p.add_argument('--no-amp', dest='amp', action='store_false')
    p.add_argument('--grad-ckpt', dest='grad_ckpt', action='store_true', default=None,
                   help='gradient checkpointing: -30%% VRAM, +20%% tiempo '
                        '(auto-on en xl/xxl)')
    p.add_argument('--no-grad-ckpt', dest='grad_ckpt', action='store_false')
    # --- exploracion ---
    p.add_argument('--epsilon-start', type=float, default=1.0)
    p.add_argument('--epsilon-floor', type=float, default=0.10)
    p.add_argument('--epsilon-decay', type=float, default=0.9999)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--antistuck', action='store_true', default=True)
    p.add_argument('--no-antistuck', dest='antistuck', action='store_false')
    # --- ckpt ---
    p.add_argument('--checkpoint-dir', default='checkpoints_single')
    p.add_argument('--device', default=None)
    p.add_argument('--from-scratch', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--metrics-csv', default=None,
                   help='CSV acumulativo de curvas (default <ckpt>/metrics_single.csv)')
    args = p.parse_args()

    store = CheckpointStore(args.checkpoint_dir)
    csv_path = Path(args.metrics_csv) if args.metrics_csv else Path(args.checkpoint_dir) / 'metrics_single.csv'
    tracker = MetricsTracker(csv_path)
    auto_bs = {'micro': 64, 'small': 64, 'medium': 64, 'large': 32,
               'gpt2': 32, 'xl': 16, 'xxl': 4}
    batch_size = args.batch_size or auto_bs.get(args.model_size, 64)
    # Defaults de memoria: los modelos grandes no caben en 6GB con Adam (2
    # estados/param). Si el usuario no lo fijo explicitamente, xl/xxl usan
    # adafactor + amp + grad_ckpt, que es lo unico que entra. Cualquier flag
    # explicito (--optimizer, --amp/--no-amp, --grad-ckpt/--no-grad-ckpt) gana.
    big_model = args.model_size in ('xl', 'xxl')
    optimizer = args.optimizer or ('adafactor' if big_model else 'adam')
    amp = big_model if args.amp is None else args.amp
    grad_ckpt = big_model if args.grad_ckpt is None else args.grad_ckpt
    agent = TopoMarioAgent(device=args.device, store=store, lr=args.lr,
                           gamma=args.gamma, gae_lambda=args.gae_lambda,
                           clip=args.clip, value_coef=args.value_coef,
                           entropy_coef=args.entropy_coef,
                           min_entropy=args.min_entropy,
                           epochs=args.epochs, batch_size=batch_size,
                            seed=args.seed, model_size=args.model_size,
                            optimizer=optimizer, amp=amp,
                            grad_ckpt=grad_ckpt,
                            cause_coef=args.cause_coef,
                            micro_bs=args.micro_bs)
    agent.config.AE_RECON_WEIGHT = args.ae_weight
    print(f'[mem] optimizer={optimizer} amp={amp} '
          f'grad_ckpt={grad_ckpt} batch={batch_size} epochs={args.epochs}')

    resume_from = 0
    epsilon = args.epsilon_start
    ep0 = 0
    try:
        if store.exists() and not args.from_scratch:
            resume_from = agent.load()
            meta = store.read_state() or {}
            old = meta.get('model_size')
            if old and old != args.model_size:
                print(f'[warn] checkpoint era {old}, ahora {args.model_size}: '
                      f'reuso parcial (strict=False). Si va mal, usa --from-scratch.')
            old_opt = meta.get('optimizer')
            if old_opt and old_opt != optimizer:
                print(f'[warn] checkpoint usaba optimizer {old_opt}, ahora '
                      f'{args.optimizer}: momento fresco, pesos se conservan.')
            # acumulativo: epsilon y contador siguen donde quedaron
            if 'epsilon' in meta:
                try:
                    epsilon = max(args.epsilon_floor,
                                  min(float(meta['epsilon']), args.epsilon_start))
                except (TypeError, ValueError):
                    pass
            try:
                ep0 = int(meta.get('episode', 0))
            except (TypeError, ValueError):
                ep0 = 0
            print(f'resumed from {store.slot} @ step {resume_from} '
                  f'(model={args.model_size} eps={epsilon:.3f} ep={ep0})')
        else:
            agent.acquire_lock()
            print(f'starting from scratch (model={args.model_size})')
    except CheckpointLockError as exc:
        print(f'ERROR: {exc}')
        return 1

    n_params = sum(p.numel() for p in agent.model.parameters())
    print(f'device={agent.device} model={args.model_size} params={n_params:,} '
          f'layers={agent.config.N_LAYERS} d={agent.config.D_MODEL} '
          f'spectral={agent.config.NUM_SPECTRAL_LAYERS} '
          f'torus={agent.config.TORUS_RADIAL_BINS}x{agent.config.TORUS_ANGULAR_BINS}')

    env = create_environment(stage=args.stage, frame_skip=args.frameskip,
                             max_steps=args.max_steps,
                             stuck_limit=args.stuck_limit,
                             inactivity_limit=args.inactivity_limit)
    names = env.action_names
    dash = Dashboard(names, enabled=not args.no_viz, tracker=tracker)

    recent = deque(maxlen=20)
    metrics: dict = {}
    updates = [0]
    banked = [0]
    rewards, progresses = list(tracker.ep_reward), list(tracker.ep_max_x)
    mixes: list = []

    def current_metrics():
        return dict(metrics, episodes_done=len(recent),
                    epsilon=epsilon, deaths=env.deaths,
                    levels_completed=env.levels_completed,
                    model_size=args.model_size)

    install_save_on_signal(agent, current_metrics)

    def flush():
        if args.no_learn or agent.banked() < args.rollout:
            return
        out = agent.learn()
        if not out:
            return
        metrics.clear()
        metrics.update(out)
        metrics.update(current_metrics())
        updates[0] += 1
        mixes.append(out['action_mix'])
        tracker.log_update(out['policy_loss'], out['value_loss'],
                           out['entropy'], out['approx_kl'])
        print(f'  upd#{updates[0]} step {agent.step} | '
              f'pl {out["policy_loss"]:+.4f} vl {out["value_loss"]:8.3f} '
              f'H {out["entropy"]:.4f} kl {out["approx_kl"]:.5f} | '
              f'causa L={out.get("cause_loss", 0.0):.4f} '
              f'acc={out.get("cause_acc", 0.0):.2f} | '
              f'mix {out["action_mix"]}')
        if args.save_every and updates[0] % args.save_every == 0:
            agent.save(metrics)

    ep = ep0
    try:
        while True:
            ep += 1
            if args.episodes and ep > args.episodes:
                break
            screen, semantic = env.reset()
            info: dict = {}
            ep_reward = 0.0
            max_x = 0
            steps = 0
            counts = np.zeros(len(names), dtype=int)
            bd_sum = {k: 0.0 for k in
                      MetricsTracker.POS_KEYS + MetricsTracker.NEG_KEYS}
            done = False
            # PORQUE del paso anterior: tercer modo de entrada (16 causas).
            # Se actualiza en el momento exacto que el env informa la causa.
            prev_cause = np.zeros(16, dtype=np.float32)

            while steps < args.max_steps and not done:
                force = (2 if (args.antistuck and info.get('stuck_steps', 0) > 5)
                         else None)
                (action, logp, value, source), logits = agent.act(
                    screen, semantic, epsilon=epsilon, force=force,
                    return_logits=True, prev_cause=prev_cause)
                agent.observe(screen, semantic, action, logp, 0.0, False, value,
                              prev_cause=prev_cause)
                screen, semantic, reward, done, info = env.step(action)
                agent.buffer.reward[-1] = reward
                agent.buffer.done[-1] = float(done)
                # el PORQUE se fija en el momento exacto del premio/castigo,
                # no como amalgama al final del episodio
                cause_vec = info.get('cause_vec')
                if cause_vec is None:
                    cause_vec = np.zeros(16, dtype=np.float32)
                else:
                    cause_vec = np.asarray(cause_vec, dtype=np.float32)
                agent.buffer.cause[-1] = cause_vec
                banked[0] += 1
                for k in bd_sum:
                    bd_sum[k] += float(info.get('reward_breakdown', {}).get(k, 0.0))

                steps += 1
                ep_reward += reward
                max_x = max(max_x, info.get('progress_x', 0))
                counts[action] += 1
                # log del momento exacto: que causa disparo y cuanto pago
                important = float(cause_vec[[1, 2, 5, 6, 7, 8, 11, 12, 13, 14]].sum())
                if important > 0:
                    print(f'    [CAUSA] t{steps} a={names[action]} '
                          f'r={reward:+.2f} {info.get("cause_text", "")}')
                prev_cause = cause_vec
                epsilon = max(args.epsilon_floor, epsilon * args.epsilon_decay)

                if not args.no_viz:
                    dash.update(screen, get_focus(agent, screen),
                                logits.cpu().numpy(), info, action, epsilon)
                flush()

                if steps % 200 == 0:
                    print(f'  ep{ep} tick{steps:5d} r={ep_reward:8.1f} '
                          f'x={max_x:5d}/{info.get("goal_target", 0):5d} '
                          f'src={source} eps={epsilon:.3f} stuck={info.get("stuck_steps", 0)}')

            reason = next((k for k in ('level_complete', 'died', 'stuck',
                                       'inactive', 'timeout') if info.get(k)), 'unknown')
            recent.append(float(ep_reward))
            rewards.append(float(ep_reward))
            progresses.append(int(max_x))
            acc = float(info.get('goal_success_rate',
                                 env.shaper.goal.success_rate))
            pos = sum(bd_sum[k] for k in MetricsTracker.POS_KEYS)
            neg = sum(bd_sum[k] for k in MetricsTracker.NEG_KEYS)
            tracker.log_episode(ep_reward, max_x, acc, epsilon, pos, neg)
            flush()
            agent.save({**current_metrics(), 'episode_reward': float(ep_reward),
                        'max_x': int(max_x), 'frames': steps * args.frameskip,
                        'end': reason, 'episode': ep})
            last = metrics or {}
            mix_now = (np.array(mixes).mean(axis=0).round(3).tolist()
                       if mixes else (counts / max(1, counts.sum())).round(3).tolist())
            print(f'ep{ep}: reward={ep_reward:.1f} ticks={steps} '
                  f'max_x={max_x} end={reason} '
                  f'dist={np.round(counts / max(1, counts.sum()), 2)} '
                  f'banked={banked[0]} eps={epsilon:.3f}')
            # bloque detallado estilo neuromario para decidir desarrollo
            avg10 = float(np.mean(rewards[-10:])) if rewards else 0.0
            print('=' * 60)
            print(f'[ep{ep}] reward={ep_reward:.1f} avg10={avg10:.1f} '
                  f'max_x={max_x} acc_metas={acc:.2f} eps={epsilon:.3f}')
            print(f"  loss: pl={last.get('policy_loss', 0.0):+.4f} "
                  f"vl={last.get('value_loss', 0.0):8.3f} "
                  f"H={last.get('entropy', 0.0):.4f} "
                  f"kl={last.get('approx_kl', 0.0):.5f}")
            plus = ' '.join(f'{k}={bd_sum[k]:+.1f}'
                            for k in MetricsTracker.POS_KEYS if bd_sum[k])
            minus = ' '.join(f'{k}={bd_sum[k]:+.1f}'
                             for k in MetricsTracker.NEG_KEYS if bd_sum[k])
            print(f'  +suma ({pos:+.1f}): {plus or "-"}')
            print(f'  -resta ({neg:+.1f}): {minus or "-"}')
            print(f'  mix={mix_now} deaths={env.deaths} '
                  f'flags={env.levels_completed} csv={csv_path}')
            print('=' * 60)
            dash.update_metrics(
                mix=mix_now, breakdown=bd_sum,
                extra_text=f'ep{ep} r={ep_reward:.0f} x={max_x} '
                           f'deaths={env.deaths} flags={env.levels_completed}\n'
                           f"mix={dict(zip(names, np.array(mix_now).round(2)))}")
    finally:
        env.close()
        dash.close()
        agent.save({**current_metrics(), 'final': True})

    mix = np.zeros(len(names))
    print(f'ppo updates: {updates[0]} banked={banked[0]}')
    print(f'mean_reward={np.mean(rewards):.1f} best_x={max(progresses) if progresses else 0} '
          f'ckpt={store.slot} @ {agent.step} deaths={env.deaths} flags={env.levels_completed}')


if __name__ == '__main__':
    sys.exit(main() or 0)
