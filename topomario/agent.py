#!/usr/bin/env python3
"""
Shared agent + checkpoint store for TopoMario.

Both ``train.py`` and ``play.py`` drive the same ``TopoMarioAgent`` so that
whichever entry point you use, the policy learns and lands in exactly one
checkpoint slot (``<root>/last``). Saving is atomic (temp dir + rename) and
prunes every other checkpoint directory, so progress is never lost and the
checkpoint tree never fills up with junk.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file as st_save
from safetensors.torch import load_file as st_load
from torch.distributions import Categorical

from .model import TopoMario, TopoMarioConfig


class CheckpointLockError(RuntimeError):
    """Another process already owns this checkpoint directory."""


class CheckpointLock:
    """Exclusive advisory lock on a checkpoint root.

    Two trainers sharing one directory interleave their atomic swaps and destroy
    the slot: process A renames slot->backup while process B is doing the same,
    and one of them ends up with a directory that has meta.json but no weights.
    The history file makes this visible (interleaved pids), and the fix is to
    refuse the second writer outright instead of racing.
    """

    def __init__(self, root: Path):
        self.path = Path(root) / '.lock'
        self.fh = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, 'a+')
        try:
            import fcntl
            fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except ImportError:
            return  # non-POSIX: skip locking
        except OSError as exc:
            self.fh.seek(0)
            holder = self.fh.read().strip() or 'unknown'
            self.fh.close()
            self.fh = None
            raise CheckpointLockError(
                f'{self.path.parent} is locked by pid {holder}. '
                f'Use a different --checkpoint-dir, or stop that process first.'
            ) from exc
        self.fh.seek(0)
        self.fh.truncate()
        self.fh.write(str(os.getpid()))
        self.fh.flush()

    def release(self) -> None:
        if self.fh is None:
            return
        try:
            import fcntl
            fcntl.flock(self.fh, fcntl.LOCK_UN)
        except Exception:
            pass
        self.fh.close()
        self.fh = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()


class CheckpointStore:
    """Single-slot, atomic, pruning checkpoint store.

    Safety rules, because losing a long run to a stray ``rm -rf`` or a buggy
    prune is unacceptable:

    - ``prune()`` only ever touches paths this store itself created: the slot
      directory, its temp/backup siblings, and top-level files matching the
      checkpoint extensions. Anything else in the root is left alone.
    - the previous slot is kept as ``last.bak`` (pruned to one generation) so a
      corrupted write can still be recovered.
    - nothing here is ever removed on ``__init__``; call ``prune()`` explicitly.
    """

    SLOT_FILES = ('model.safetensors', 'optimizer.pt', 'meta.json')

    def __init__(self, root: str = 'checkpoints_topomario', slot: str = 'last'):
        self.root = Path(root)
        self.slot_name = slot
        self.slot = self.root / slot
        self.backup = self.root / f'{slot}.bak'
        self.history = self.root / 'topomario_history.jsonl'
        self.tmp = self.root / f'.{slot}.tmp'

    def save(self, model: nn.Module, optimizer: Optional[torch.optim.Optimizer],
             step: int, metrics: Optional[Dict] = None) -> Path:
        metrics = metrics or {}
        self.root.mkdir(parents=True, exist_ok=True)

        # PID-scoped temp dir: two processes can never collide on it
        tmp = self.root / f'.{self.slot_name}.tmp.{os.getpid()}'
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)

        sd = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
        st_save(sd, str(tmp / 'model.safetensors'))

        if optimizer is not None:
            torch.save(optimizer.state_dict(), str(tmp / 'optimizer.pt'))

        payload = {'step': int(step), 'pid': os.getpid(), **metrics}
        (tmp / 'meta.json').write_text(json.dumps(payload, indent=2))

        # verify the staged slot is complete BEFORE swapping it in
        staged_ok = (tmp / 'model.safetensors').exists() and \
                    (tmp / 'meta.json').exists()
        if not staged_ok:
            shutil.rmtree(tmp, ignore_errors=True)
            raise RuntimeError('staged checkpoint incomplete; slot left untouched')

        self._replace_slot(tmp)
        self._append_history(payload)
        return self.slot

    def _replace_slot(self, tmp: Path) -> None:
        """Swap tmp into the slot, keeping the previous generation as backup."""
        backup = self.backup
        if backup.exists():
            # not ignore_errors: a silently-failed rmtree is what left a
            # half-populated slot behind before
            shutil.rmtree(backup)
        if self.slot.exists():
            self.slot.rename(backup)
        try:
            tmp.rename(self.slot)
        except OSError as exc:
            # roll the previous slot back rather than leaving nothing usable
            if not self.slot.exists() and backup.exists():
                backup.rename(self.slot)
            raise RuntimeError(f'failed to install checkpoint: {exc}') from exc

    def load(self, model: nn.Module,
             optimizer: Optional[torch.optim.Optimizer] = None) -> int:
        """Load model (+optimizer) from the slot. Returns the saved step.

        Falls back to ``last.bak`` if the primary slot is missing or unreadable.
        Uses strict=False so upgraded encoders (saliency/gate/decoder de
        MarioVision) reutilizan los pesos viejos y arrancan aleatorio solo lo
        nuevo; tambien permite cambiar --model-size sin romper el resume.
        """
        for source in (self.slot, self.backup):
            model_file = source / 'model.safetensors'
            if not model_file.exists():
                continue
            try:
                model.load_state_dict(st_load(str(model_file)), strict=True)
            except Exception as exc:  # corrupt write, arch upgrade o size distinto
                try:
                    missing, unexpected = model.load_state_dict(
                        st_load(str(model_file)), strict=False), None
                    print(f'[checkpoint] {source} partial load ({exc}); '
                          f'missing={len(missing[0]) if missing else "?"} '
                          f'unexpected={len(missing[1]) if missing else "?"}')
                except Exception as exc2:
                    print(f'[checkpoint] {source} unreadable ({exc2}); trying backup')
                    continue

            if optimizer is not None:
                opt_file = source / 'optimizer.pt'
                if opt_file.exists():
                    try:
                        optimizer.load_state_dict(torch.load(str(opt_file),
                                                             map_location='cpu',
                                                             weights_only=False))
                    except Exception as exc_opt:
                        # cambio de optimizer (adam->adafactor) o de tamano:
                        # se sigue con momento fresco en vez de romper el resume
                        print(f'[checkpoint] optimizer state no reutilizado '
                              f'({exc_opt}); momento fresco')
            meta_file = source / 'meta.json'
            if meta_file.exists():
                return int(json.loads(meta_file.read_text()).get('step', 0))
            return 0
        return 0

    def exists(self) -> bool:
        return ((self.slot / 'model.safetensors').exists()
                or (self.backup / 'model.safetensors').exists())

    def read_state(self) -> Optional[Dict]:
        """Read meta.json from the slot (or backup) without touching weights."""
        for source in (self.slot, self.backup):
            meta = source / 'meta.json'
            if meta.exists():
                try:
                    return json.loads(meta.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
        return None

    def _append_history(self, payload: Dict) -> None:
        try:
            with open(self.history, 'a') as fh:
                fh.write(json.dumps(payload) + '\n')
        except OSError:
            pass

    def prune(self) -> List[str]:
        """Remove checkpoints this store owns, keeping the slot + one backup."""
        removed = []
        if not self.root.exists():
            return removed
        keep = {self.slot_name, self.backup.name}
        for child in self.root.iterdir():
            if child.is_dir() and child.name not in keep \
                    and (child.name.startswith(f'.{self.slot_name}.')
                         or child.name.startswith('step_')):
                shutil.rmtree(child, ignore_errors=True)
                removed.append(child.name)
            elif child.is_file() and child.suffix in ('.safetensors', '.pt'):
                child.unlink(missing_ok=True)
                removed.append(child.name)
        return removed


def stratified_actions(probs: np.ndarray, rng) -> np.ndarray:
    """Pick one action per row so a batch never collapses onto a single action.

    Without this, N identical NES envs reset to the same state and a near-greedy
    policy all press the same button: the batch becomes N copies of one
    trajectory, so N times the compute buys almost no extra gradient signal.

    Allocation is proportional to the mean policy, so the batch's empirical
    action mix stays on-policy, but coverage is guaranteed: every action with a
    nonzero quota appears. Each row still takes its own most-preferred action
    that is still available.

    Returns real action indices in ``[0, n_actions)``.
    """
    n_rows, n_actions = probs.shape
    mean_p = probs.mean(axis=0)
    counts = np.floor(mean_p * n_rows).astype(int)

    # largest-remainder fill so the counts sum to exactly n_rows
    short = n_rows - int(counts.sum())
    if short > 0:
        order = np.argsort(-(mean_p * n_rows - counts))
        counts[order[:short]] += 1

    pool = [a for a, c in enumerate(counts) for _ in range(int(c))]
    rng.shuffle(pool)

    out = np.empty(n_rows, dtype=np.int64)
    for i in range(n_rows):
        if not pool:
            out[i] = rng.integers(n_actions)
            continue
        for a in np.argsort(-probs[i]):
            if a in pool:
                pool.remove(int(a))
                out[i] = a
                break
    return out


class ValueScaler:
    """Running mean/std of discounted returns.

    The critic was diverging (value_loss 945-2771 against policy_loss 0.02), so
    with a shared grad-norm clip every update spent its whole budget fitting the
    critic and wrecked the policy. Predicting *normalized* returns keeps the
    value loss O(1) so the policy gradient is not swamped.

    Returns from the reward shaper are in the hundreds, so the first batch would
    otherwise land as a 50k loss spike and destroy the weights. Until enough
    samples are banked, targets are divided by a fixed conservative constant
    instead of the running std.
    """

    WARMUP_SAMPLES = 256
    INIT_STD = 100.0

    def __init__(self, beta: float = 0.999):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4
        self.beta = beta

    def update(self, returns: np.ndarray) -> None:
        if returns.size == 0:
            return
        batch_mean = float(returns.mean())
        batch_var = float(returns.var())
        batch_count = float(returns.size)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total
        self.mean = new_mean
        self.var = max(1e-8, m2 / total)
        self.count = total

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var))

    def scale(self, returns: np.ndarray) -> np.ndarray:
        """Normalize returns, falling back to a fixed scale while warming up."""
        if self.count < self.WARMUP_SAMPLES:
            return (returns / self.INIT_STD).astype(np.float32)
        return ((returns - self.mean) / self.std).astype(np.float32)

    def state(self) -> Dict:
        return {'mean': self.mean, 'var': self.var, 'count': self.count}


class RolloutBuffer:
    N_CAUSES = 16

    def __init__(self):
        self.batched = False
        self.clear()

    def clear(self):
        self.screen: List[np.ndarray] = []
        self.sem: List[np.ndarray] = []
        self.action: List[int] = []
        self.logp: List[float] = []
        self.reward: List[float] = []
        self.done: List[float] = []
        self.value: List[float] = []
        # PORQUE del paso: vector causa [16] binario (que premio/castigo
        # disparo). Sin esto el modelo solo ve el escalar y no sabe que
        # aprender. Se supervisa con cabeza auxiliar (BCE) en learn().
        self.cause: List[np.ndarray] = []
        self.prev_cause: List[np.ndarray] = []

    def add(self, screen, sem, action, logp, reward, done, value,
            cause=None, prev_cause=None):
        """Store a transition.

        Accepts either one env (scalar fields) or a vectorized batch of N envs,
        where ``screen`` is [N,3,H,W] and the scalar fields are length-N arrays.
        ``stacked()`` flattens both cases back to a leading batch dimension.
        ``cause``/``prev_cause``: [16] o [N,16]; None -> ceros.
        """
        self.screen.append(screen)
        self.sem.append(sem)
        self.action.append(action)
        self.logp.append(logp)
        self.reward.append(reward)
        self.done.append(done)
        self.value.append(value)
        # causa: normaliza a array para que stacked() lo apile igual que sem
        try:
            import numpy as _np
            z16 = _np.zeros(self.N_CAUSES, dtype=_np.float32)
            if cause is None:
                # infiere forma batched/single por el screen
                s = _np.asarray(screen)
                if self.batched and s.ndim == 5:
                    cause = _np.zeros((s.shape[0], self.N_CAUSES),
                                      dtype=_np.float32)
                else:
                    cause = z16
            if prev_cause is None:
                prev_cause = ( _np.zeros_like(_np.asarray(cause, dtype=_np.float32))
                               if not isinstance(cause, list) else z16)
            self.cause.append(_np.asarray(cause, dtype=_np.float32))
            self.prev_cause.append(_np.asarray(prev_cause, dtype=_np.float32))
        except Exception:
            pass

    def __len__(self):
        return len(self.action)

    def stacked(self, name: str, device: str = 'cpu') -> torch.Tensor:
        """Flatten a possibly-batched buffer field into a leading-batch tensor.

        The two cases are indistinguishable by rank alone -- a single-env
        [T,3,240,256] and a batched [T,N,3,240,256] both look like ndim>=2 -- so
        ``batched`` must be stated by the caller. Getting it wrong reshapes
        [T,3,240,256] into [T*3,240,256], which then feeds 64 channels into a
        3-channel conv.
        """
        arr = np.asarray(getattr(self, name))
        if name == 'screen':
            # [T,N,3,H,W] -> [T*N,3,H,W]  /  [T,3,H,W] -> [T,3,H,W]
            arr = (arr.reshape((-1,) + arr.shape[2:]) if self.batched
                   else arr)
        elif name in ('sem', 'cause', 'prev_cause'):
            # [T,N,D] -> [T*N,D]  /  [T,D] -> [T,D]
            arr = (arr.reshape((-1, arr.shape[-1])) if self.batched
                   else arr)
        else:
            arr = arr.reshape(-1)
        return torch.from_numpy(np.ascontiguousarray(arr)).float().to(device)


class TopoMarioAgent:
    """Policy + critic + PPO update + checkpointing. Shared by train and play."""

    def __init__(self,
                 device: Optional[str] = None,
                 store: Optional[CheckpointStore] = None,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.02,
                 min_entropy: float = 0.8,
                 entropy_floor_coef: float = 0.5,
                 log_ratio_clamp: float = 0.2,
                 max_grad_norm: float = 0.5,
                 epochs: int = 4,
                 batch_size: int = 64,
                 seed: int = 0,
                 model_size: str = 'small',
                 optimizer: str = 'adam',
                 amp: bool = False,
                 grad_ckpt: bool = False,
                 cause_coef: float = 0.5,
                 value_clip: float = 0.2,
                 micro_bs: int = 0):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = TopoMarioConfig(SCALE=model_size)
        self.model_size = model_size
        self.config.DEVICE = self.device
        if grad_ckpt:
            self.config.GRADIENT_CHECKPOINTING = True
        self.model = TopoMario(self.config).to(self.device)
        self.optimizer_name = optimizer
        self.lr = lr
        self.optimizer = self._make_optimizer(optimizer, lr)
        self.amp = amp and 'cuda' in self.device
        self.scaler_amp = torch.amp.GradScaler('cuda') if self.amp else None

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip = clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.min_entropy = min_entropy
        self.entropy_floor_coef = entropy_floor_coef
        self.log_ratio_clamp = log_ratio_clamp
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs
        self.batch_size = batch_size
        self.cause_coef = float(cause_coef)
        self.value_clip = float(value_clip)
        # micro-batch CUDA (0 = auto min(batch,8)). xxl necesita 1-2.
        self.micro_bs = int(micro_bs or 0)
        self.store = store or CheckpointStore()

        self.buffer = RolloutBuffer()
        self.scaler = ValueScaler()
        self.step = 0
        self.rng = np.random.default_rng(seed)
        self.epsilon = 1.0
        self._lock = CheckpointLock(Path(store.root) if store
                                    else Path('checkpoints_topomario'))

    def _make_optimizer(self, name: str, lr: float):
        """adam = 2 estados/param (~4.7GB en xxl); sgd = 1 estado;
        adafactor = segundo momento factorizado (filas+cols, no NxM): el mas
        ligero, cabe en 6GB donde Adam no. Para xxl es el unico que entra."""
        name = (name or 'adam').lower()
        params = self.model.parameters()
        if name == 'sgd':
            return torch.optim.SGD(params, lr=lr * 10, momentum=0.9, nesterov=True)
        if name == 'adafactor':
            # torch.optim.Adafactor (>=2.5) usa la firma nativa, NO la de
            # HuggingFace. Pasarle clip_threshold/decay_rate/beta1/... lanza
            # TypeError y antes caia a Adam -> OOM. Detectamos la firma real.
            import inspect
            if hasattr(torch.optim, 'Adafactor'):
                try:
                    sig = inspect.signature(torch.optim.Adafactor)
                    if 'beta2_decay' in sig.parameters:  # API nativa torch
                        return torch.optim.Adafactor(params, lr=lr)
                    # API estilo HuggingFace (torch antiguo o backport)
                    return torch.optim.Adafactor(
                        params, lr=lr, eps=(1e-30, 1e-3), clip_threshold=1.0,
                        decay_rate=-0.8, beta1=None, weight_decay=0.0,
                        scale_parameter=True, relative_step=False,
                        warmup_init=False)
                except Exception as exc:
                    print(f'[opt] Adafactor fallo ({exc}); caigo a SGD '
                          f'(1 estado, cabe donde Adam no)')
                    return torch.optim.SGD(params, lr=lr * 10, momentum=0.9,
                                           nesterov=True)
            print('[opt] torch sin Adafactor; uso SGD (mas ligero que Adam)')
            return torch.optim.SGD(params, lr=lr * 10, momentum=0.9,
                                   nesterov=True)
        return torch.optim.Adam(params, lr=lr, eps=1e-5)

    # -- persistence -----------------------------------------------------
    def load(self) -> int:
        """Take the checkpoint lock, then resume. Refuses a second writer."""
        self._lock.acquire()
        self.step = self.store.load(self.model, self.optimizer)
        state = self.store.read_state()
        if state:
            for key in ('value_mean', 'value_var', 'value_count'):
                if key in state:
                    setattr(self.scaler,
                            {'value_mean': 'mean', 'value_var': 'var',
                             'value_count': 'count'}[key], state[key])
        return self.step

    def acquire_lock(self) -> None:
        """Take the lock without loading (for --from-scratch runs)."""
        self._lock.acquire()

    def save(self, metrics: Optional[Dict] = None) -> None:
        payload = dict(metrics or {})
        payload.update({
            'model_size': getattr(self, 'model_size', self.config.SCALE),
            'optimizer': getattr(self, 'optimizer_name', 'adam'),
            'value_mean': self.scaler.mean,
            'value_var': self.scaler.var,
            'value_count': self.scaler.count,
            'value_std': self.scaler.std,
        })
        self.store.save(self.model, self.optimizer, self.step, payload)

    # -- acting ----------------------------------------------------------
    def act(self, screen: np.ndarray, semantic: np.ndarray,
            epsilon: float = 0.0,
            force: Optional[int] = None,
            return_logits: bool = False,
            prev_cause: Optional[np.ndarray] = None):
        """Pick an action and return (action, logp, value, source[, logits]).

        ``logp`` is always the log-prob of the action *actually taken* under the
        current policy, including when epsilon exploration or an anti-stuck
        override forces it. Otherwise the PPO ratio compares incompatible
        distributions and the update is garbage.
        ``prev_cause``: vector [16] con el PORQUE del paso anterior; entra
        como tercer modo al encoder para condicionar la decision.
        """
        self.model.eval()
        with torch.no_grad():
            s = torch.from_numpy(screen).float().unsqueeze(0).to(self.device)
            q = torch.from_numpy(semantic).float().unsqueeze(0).to(self.device)
            pc = (torch.from_numpy(np.asarray(prev_cause, dtype=np.float32))
                  .float().unsqueeze(0).to(self.device)
                  if prev_cause is not None else None)
            logits, value, _ = self.model(s, q, pc)
            log_probs = torch.log_softmax(logits, dim=-1)
            n = self.model.config.N_ACTIONS

            if force is not None:
                action, source = int(force), 'forced'
            elif self.rng.random() < epsilon:
                action, source = int(self.rng.integers(n)), 'eps'
            else:
                action, source = int(logits.argmax().item()), 'model'

            logp = float(log_probs[0, action].item())

        out = (action, logp, float(value.item()), source)
        if return_logits:
            return out, logits[0]
        return out

    def act_batch(self, screens: np.ndarray, semantics: np.ndarray,
                  epsilon: float = 0.0,
                  force: Optional[np.ndarray] = None,
                  diverse: bool = True,
                  temperature: float = 1.0,
                  prev_cause: Optional[np.ndarray] = None):
        """Act on a batch of states. ``force`` is an int array or None.

        ``diverse`` (default) uses stratified sampling so the batch covers
        several buttons instead of N copies of the greedy one.

        Returns (actions, logp, value, sources, logits).
        """
        self.model.eval()
        with torch.no_grad():
            s = torch.from_numpy(screens).float().to(self.device)
            q = torch.from_numpy(semantics).float().to(self.device)
            pc = (torch.from_numpy(np.asarray(prev_cause, dtype=np.float32))
                  .float().to(self.device)
                  if prev_cause is not None else None)
            logits, value, _ = self.model(s, q, pc)
            log_probs = torch.log_softmax(logits, dim=-1)

            probs = torch.softmax(logits / max(temperature, 1e-3),
                                  dim=-1).cpu().numpy()

            if diverse and len(screens) > 1:
                actions = stratified_actions(probs, self.rng)
                sources = np.full(len(screens), 'diverse')
            else:
                explore = self.rng.random(len(screens)) < epsilon
                actions = probs.argmax(axis=1)
                rand = self.rng.integers(0, self.model.config.N_ACTIONS,
                                         size=len(screens))
                actions = np.where(explore, rand, actions)
                sources = np.where(explore, 'eps', 'model')

            if force is not None:
                # override only the rows actually marked; -1 means "no override".
                # Doing this *after* the diverse pick keeps the other envs
                # decorrelated -- short-circuiting on `force is not None` made
                # every env fall back to argmax, i.e. 8 identical trajectories.
                forced = np.asarray(force)
                mask = forced >= 0
                if mask.any():
                    actions = np.where(mask, forced, actions)
                    sources = np.where(mask, 'forced', sources)

            act_t = torch.from_numpy(actions.astype(np.int64)).to(logits.device)
            logp = log_probs.gather(1, act_t.unsqueeze(1)).squeeze(1)

        return actions, logp.cpu().numpy(), value.cpu().numpy(), sources, logits

    def observe_batch(self, screens, semantics, actions, logp, values,
                        prev_cause=None):
        self.buffer.batched = True
        self.buffer.add(screens, semantics, actions, logp, np.zeros(len(actions)),
                        np.zeros(len(actions)), values,
                        cause=None, prev_cause=prev_cause)
        self.step += len(actions)

    # -- learning --------------------------------------------------------
    def observe(self, screen, semantic, action, logp, reward, done, value,
                cause=None, prev_cause=None):
        self.buffer.batched = False
        self.buffer.add(screen, semantic, action, logp, reward, done, value,
                         cause=cause, prev_cause=prev_cause)
        self.step += 1

    def ready(self, rollout: int) -> bool:
        return self.banked() >= rollout

    def banked(self) -> int:
        """Transitions waiting in the buffer (envs x ticks), not ticks.

        ``len(buffer)`` counts timesteps, which under-counts by a factor of N for
        a vectorized rollout and made ``--min-rollout`` unreachable.
        """
        if not len(self.buffer):
            return 0
        first = self.buffer.action[0]
        return len(self.buffer) * (len(first) if hasattr(first, '__len__') else 1)

    def learn(self) -> Optional[Dict]:
        """PPO update over the buffer. Handles both single and batched rollouts.

        OOM-safe: si CUDA se queda sin memoria (xxl en 6GB), vacia cache y
        reintenta con la mitad de batch hasta 8, en vez de matar la noche de
        entrenamiento. Con --amp (autocast+scaler) el pico baja ~40%.
        """
        if not len(self.buffer):
            return None

        rewards = np.array(self.buffer.reward, dtype=np.float32)
        dones = np.array(self.buffer.done, dtype=np.float32)
        values = np.array(self.buffer.value, dtype=np.float32)
        batched = rewards.ndim == 2  # [T, N] per-tick batch of N envs
        if not batched:
            # single env: store as [1, T] so one code path handles both
            rewards = rewards[None, :]
            dones = dones[None, :]
            values = values[None, :]

        T = rewards.shape[0]
        N = rewards.shape[1]
        adv = np.zeros_like(rewards)
        # GAE runs along the time axis independently per env
        for b in range(N):
            last = 0.0
            for t in reversed(range(T)):
                future = values[t + 1, b] if t + 1 < T else last
                delta = (rewards[t, b] + self.gamma * future * (1 - dones[t, b])
                         - values[t, b])
                last = adv[t, b] = (delta + self.gamma * self.gae_lambda
                                    * (1 - dones[t, b]) * adv[t, b])
        # [T, N] -> N-major flatten, matching screens.reshape(-1, ...)
        flat_adv = np.ascontiguousarray(adv.T).reshape(-1)
        returns = flat_adv + np.ascontiguousarray(values.T).reshape(-1)
        adv = flat_adv
        self.scaler.update(returns)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ret_target = self.scaler.scale(returns)

        screens = self.buffer.stacked('screen', 'cpu')
        sems = self.buffer.stacked('sem', 'cpu')
        try:
            causes = self.buffer.stacked('cause', self.device)
            prev_causes = self.buffer.stacked('prev_cause', self.device)
            if causes.shape[0] != screens.shape[0]:
                raise ValueError('cause/screen mismatch')
        except Exception:
            causes = torch.zeros((screens.shape[0], 16), device=self.device)
            prev_causes = torch.zeros((screens.shape[0], 16), device=self.device)
        acts = torch.tensor(np.asarray(self.buffer.action).reshape(-1),
                            device=self.device)
        old_logp = torch.tensor(np.asarray(self.buffer.logp, dtype=np.float32)
                                .reshape(-1), device=self.device)
        adv_t = torch.from_numpy(adv).float().to(self.device)
        ret_t = torch.from_numpy(ret_target).float().to(self.device)

        n = len(acts)
        idx = np.arange(n)
        # pos_weight por causa: progress/idle disparan siempre, secret casi
        # nunca. Sin peso la BCE aprende "todo cero" con acc mentirosa del 90%.
        try:
            c_flat = np.asarray(self.buffer.cause)
            if self.buffer.batched:
                c_flat = c_flat.reshape(-1, c_flat.shape[-1])
            pos = c_flat.sum(axis=0) + 1e-6
            neg = c_flat.shape[0] - c_flat.sum(axis=0) + 1e-6
            pw = np.clip(neg / pos, 1.0, 20.0).astype(np.float32)
        except Exception:
            pw = np.ones(16, dtype=np.float32)
        cause_pw = torch.from_numpy(pw).to(self.device)
        # valores viejos para el value-clip PPO (el critico sin clip diverge
        # y se come todo el gradiente: vl 945-2771 vs pl 0.02)
        old_values = torch.tensor(
            np.asarray(self.buffer.value, dtype=np.float32).reshape(-1),
            device=self.device)
        pl = vl = ent = kl = cl = 0.0
        cause_hits = 0.0
        cause_total = 0
        updates = 0
        self.model.train()
        bs = int(self.batch_size)
        # micro-batch: el rollout (512x12x240x256 = 1.5GB) se queda en CPU
        # y a CUDA sube solo el chunk del forward. xxl (625M params =
        # 2.5GB pesos + 2.5GB grads) no cabe ni con batch=8 entero.
        micro = int(self.micro_bs or min(bs, 8))
        micro = max(1, min(micro, bs))
        while True:  # reintento OOM con micro-batch cada vez menor
            try:
                for _ in range(self.epochs):
                    self.rng.shuffle(idx)
                    for start in range(0, n, bs):
                        b = idx[start:start + bs]
                        chunks = [b[i:i + micro]
                                  for i in range(0, len(b), micro)]
                        self.optimizer.zero_grad()
                        m_pl = m_vl = m_ent = m_kl = m_cl = 0.0
                        for c in chunks:
                            # solo este chunk viaja a CUDA
                            sc = screens[c].to(self.device)
                            qc = sems[c].to(self.device)
                            w = len(c) / len(b)
                            if self.amp:
                                with torch.autocast(device_type='cuda',
                                                    dtype=torch.float16):
                                    (logits, value, _,
                                     cause_logits) = self.model(
                                        sc, qc, prev_causes[c],
                                        return_cause=True)
                                    dist = Categorical(logits=logits)
                                    new_logp = dist.log_prob(acts[c])
                                    log_ratio = torch.clamp(
                                        new_logp - old_logp[c],
                                        -self.log_ratio_clamp,
                                        self.log_ratio_clamp)
                                    ratio = torch.exp(log_ratio)
                                    s1 = ratio * adv_t[c]
                                    s2 = torch.clamp(
                                        ratio, 1 - self.clip,
                                        1 + self.clip) * adv_t[c]
                                    policy = -torch.min(s1, s2).mean()
                                    v_clipped = (
                                        old_values[c]
                                        + torch.clamp(value - old_values[c],
                                                      -self.value_clip,
                                                      self.value_clip))
                                    value_loss = torch.max(
                                        F.mse_loss(value, ret_t[c],
                                                   reduction='none'),
                                        F.mse_loss(v_clipped, ret_t[c],
                                                   reduction='none')).mean()
                                    entropy = dist.entropy().mean()
                                    approx_kl = (-log_ratio).mean()
                                    ent_floor = torch.relu(
                                        self.min_entropy - entropy)
                                    cause_loss = F.binary_cross_entropy_with_logits(
                                        cause_logits, causes[c],
                                        pos_weight=cause_pw)
                                    loss = (policy
                                            + self.value_coef * value_loss
                                            - self.entropy_coef * entropy
                                            + self.entropy_floor_coef * ent_floor
                                            + self.cause_coef * cause_loss)
                                self.scaler_amp.scale(loss * w).backward()
                            else:
                                (logits, value, _,
                                 cause_logits) = self.model(
                                    sc, qc, prev_causes[c],
                                    return_cause=True)
                                dist = Categorical(logits=logits)
                                new_logp = dist.log_prob(acts[c])

                                # Hard-log-ratio clamp before exp(). Without it
                                # a stale logp from a previous policy made exp()
                                # overflow to 1e34 (pl=+5685, kl~1e34).
                                log_ratio = new_logp - old_logp[c]
                                log_ratio = torch.clamp(
                                    log_ratio, -self.log_ratio_clamp,
                                    self.log_ratio_clamp)
                                ratio = torch.exp(log_ratio)

                                s1 = ratio * adv_t[c]
                                s2 = torch.clamp(ratio, 1 - self.clip,
                                                 1 + self.clip) * adv_t[c]
                                policy = -torch.min(s1, s2).mean()
                                v_clipped = (
                                    old_values[c]
                                    + torch.clamp(value - old_values[c],
                                                  -self.value_clip,
                                                  self.value_clip))
                                value_loss = torch.max(
                                    F.mse_loss(value, ret_t[c],
                                               reduction='none'),
                                    F.mse_loss(v_clipped, ret_t[c],
                                               reduction='none')).mean()
                                entropy = dist.entropy().mean()
                                approx_kl = (-log_ratio).mean()

                                # A floor on entropy: a fixed entropy coefficient
                                # let the policy collapse to one action
                                # (H 0.087 of 1.95). Push back below the floor.
                                ent_floor = torch.relu(
                                    self.min_entropy - entropy)
                                cause_loss = F.binary_cross_entropy_with_logits(
                                    cause_logits, causes[c],
                                    pos_weight=cause_pw)
                                loss = (policy
                                        + self.value_coef * value_loss
                                        - self.entropy_coef * entropy
                                        + self.entropy_floor_coef * ent_floor
                                        + self.cause_coef * cause_loss)
                                (loss * w).backward()
                            m_pl += policy.item() * w
                            m_vl += value_loss.item() * w
                            m_ent += entropy.item() * w
                            m_kl += approx_kl.item() * w
                            try:
                                m_cl += cause_loss.item() * w
                                with torch.no_grad():
                                    pred = (cause_logits > 0).float()
                                    cause_hits += ((pred == causes[c]).float()
                                                   .sum().item())
                                    cause_total += int(causes[c].numel())
                            except Exception:
                                pass
                        if self.amp:
                            self.scaler_amp.unscale_(self.optimizer)
                            nn.utils.clip_grad_norm_(self.model.parameters(),
                                                     self.max_grad_norm)
                            torch.nn.utils.clip_grad_value_(
                                self.model.parameters(), 10.0)
                            self.scaler_amp.step(self.optimizer)
                            self.scaler_amp.update()
                        else:
                            nn.utils.clip_grad_norm_(self.model.parameters(),
                                                     self.max_grad_norm)
                            torch.nn.utils.clip_grad_value_(
                                self.model.parameters(), 10.0)
                            self.optimizer.step()
                        pl += m_pl
                        vl += m_vl
                        ent += m_ent
                        kl += m_kl
                        cl += m_cl
                        updates += 1
                break
            except torch.OutOfMemoryError:
                try:
                    self.optimizer.zero_grad()
                except Exception:
                    pass
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                if micro <= 1:
                    # ni 1 muestra cabe (xxl en ~6GB): salta el update SIN
                    # matar la corrida; el proximo rollout lo reintenta.
                    print('[learn] OOM aun con micro=1; salto este update '
                          '(prueba --model-size xl, --batch-size 4 o '
                          '--micro-bs 1)')
                    self.buffer.clear()
                    return None
                micro = max(1, micro // 2)
                print(f'[learn] OOM; reintento con micro-batch={micro}')

        actions = np.asarray(self.buffer.action).reshape(-1)
        self.buffer.clear()
        u = max(1, updates)
        dist_counts = np.bincount(actions.astype(int),
                                  minlength=self.model.config.N_ACTIONS)
        return {
            'policy_loss': pl / u,
            'value_loss': vl / u,
            'entropy': ent / u,
            'approx_kl': kl / u,
            'cause_loss': cl / u,
            'cause_acc': (cause_hits / max(1, cause_total)),
            'action_mix': (dist_counts / dist_counts.sum()).round(3).tolist(),
        }


def install_save_on_signal(agent: TopoMarioAgent, metrics_fn=None) -> None:
    """Save the current step on SIGINT/SIGTERM so Ctrl-C never loses progress."""

    def handler(signum, frame):
        metrics = metrics_fn() if metrics_fn else None
        agent.save(metrics or {'signal': int(signum)})
        print(f'\n[checkpoint] saved at step {agent.step} (signal {signum})')
        sys.exit(130 if signum == signal.SIGINT else 143)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            pass
