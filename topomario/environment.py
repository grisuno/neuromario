#!/usr/bin/env python3
"""
TopoMario environment.

Anti-stuck machinery ported from grisuno/neuromario (trimario4.py):
  - FrameSkip            Config.FRAMESKIP = 8
  - StuckMonitor         Config.STUCK_LIMIT = 100 / INACTIVITY_LIMIT = 150
  - Reward shaping       DEATH_PENALTY / STUCK_PENALTY / GOAL_REWARD / COIN_REWARD
  - Curriculum goals     GOAL_MIN_OFFSET / GOAL_MAX_OFFSET / GOAL_BUFFER_SIZE
  - Exploration decay    EXPLORATION_DECAY_RATE = 0.985
"""
import numpy as np
from typing import Tuple, Dict, List
import warnings

warnings.filterwarnings('ignore')

try:
    import gym_super_mario_bros
    from nes_py.wrappers import JoypadSpace
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False


# Causas canonicas del reward: el modelo debe saber EL PORQUE, no solo el
# escalar. Cada key de breakdown mapea a una posicion fija. Positivas 0-8,
# negativas 9-15. Se expone como vector binario (presencia) + texto corto.
CAUSE_KEYS = ['progress', 'goal', 'coin', 'score', 'vertical', 'enemy',
              'block', 'powerup', 'secret',
              'idle', 'backtrack', 'damage', 'death', 'samespot',
              'stuck', 'inactive']
N_CAUSES = len(CAUSE_KEYS)
CAUSE_POS = set(CAUSE_KEYS[:9])

# Etiquetas cortas para el log del momento exacto (sin tildes: consola Tk).
CAUSE_TAG = {'progress': 'avanza', 'goal': 'META', 'coin': 'MONEDA',
             'score': 'puntos', 'vertical': 'salta', 'enemy': 'ENEMIGO',
             'block': 'LADRILLO', 'powerup': 'HONGO', 'secret': 'SECRETO',
             'idle': 'quieto', 'backtrack': 'retrocede', 'damage': 'DANO',
             'death': 'MUERTE', 'samespot': 'MISMO-SITIO',
             'stuck': 'ATASCADO', 'inactive': 'inactivo'}


def breakdown_to_cause_vec(breakdown: Dict, thresh: float = 1e-9) -> np.ndarray:
    """Vector binario [N_CAUSES]: 1.0 si esa causa disparo este paso."""
    vec = np.zeros(N_CAUSES, dtype=np.float32)
    for i, k in enumerate(CAUSE_KEYS):
        try:
            if abs(float(breakdown.get(k, 0.0))) > thresh:
                vec[i] = 1.0
        except (TypeError, ValueError):
            pass
    return vec


def breakdown_to_text(breakdown: Dict) -> str:
    """'+MONEDA(+1.2) +avanza(+0.8) | -quieto(-1.2)': el PORQUE del paso."""
    plus, minus = [], []
    for k in CAUSE_KEYS:
        try:
            v = float(breakdown.get(k, 0.0))
        except (TypeError, ValueError):
            continue
        if abs(v) <= 1e-9:
            continue
        tag = f'{CAUSE_TAG.get(k, k)}({v:+.1f})'
        (plus if k in CAUSE_POS else minus).append(tag)
    left = ' '.join(plus) if plus else '-'
    right = ' '.join(minus) if minus else '-'
    return f'+[{left}] | -[{right}]'


class Config:
    FRAMESKIP = 8
    STUCK_LIMIT = 100
    INACTIVITY_LIMIT = 150
    MAX_AGENT_STEPS = 2000

    # Frame stack: 4 ultimos frames apilados [12,240,256]. Sin esto la red
    # no ve velocidad (un solo frame no distingue ir vs venir del enemigo).
    # Es el truco estandar DQN/PPO-Atari. El reset repite el primer frame.
    FRAME_STACK = 4

    DEATH_PENALTY = -2.0
    STUCK_PENALTY = -3.0
    IDLE_PENALTY = -0.15  # per step with zero horizontal progress
    GOAL_REWARD = 2.5
    COIN_REWARD = 1.2
    SCORE_REWARD_SCALE = 0.015
    MAX_SCORE_REWARD = 2.5
    X_PROGRESS_SCALE = 0.1

    # Discrete game events, decoded from score deltas the way NeuroMario does.
    # SMB has no event flags in `info`, so score jumps are the signal:
    #   50-500 without the 200/1000 marks -> stomped an enemy
    #   ~200 at a new x position             -> broke a brick
    #   >2000 with coins unchanged           -> mushroom / fire flower
    #   >1000 and not a 1000-multiple        -> discovered a secret exit
    ENEMY_REWARD = 1.5
    BLOCK_REWARD = 1.0
    POWERUP_REWARD = 5.0
    SECRET_REWARD = 10.0
    DAMAGE_PENALTY = -3.0
    POWERUP_SCORE_THRESHOLD = 2000

    # Castigos nuevos: morir siempre en el mismo sitio y retroceder.
    # SAME_SPOT: si muere a <RADIUS px de una muerte anterior, extra.
    # BACKTRACK: retroceder mucho en un paso (dx muy negativo) resta.
    SAME_SPOT_PENALTY = -2.0
    SAME_SPOT_RADIUS = 40
    SAME_SPOT_MEMORY = 4
    BACKTRACK_SCALE = 0.05
    BACKTRACK_THRESHOLD = -8.0

    GOAL_MIN_OFFSET = 30
    GOAL_MAX_OFFSET = 120
    GOAL_BUFFER_SIZE = 50

    EXPLORATION_DECAY_RATE = 0.985

    ACTION_NAMES = ['NOOP', 'RIGHT', 'RIGHT_A', 'RIGHT_B', 'RIGHT_A_B', 'A', 'LEFT']
    N_ACTIONS = len(ACTION_NAMES)


class FrameSkip:
    def __init__(self, env, skip: int = Config.FRAMESKIP):
        self.env = env
        self.skip = skip

    def step(self, action: int):
        total_reward = 0.0
        done = False
        info = {}
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            done = terminated or truncated
            if done:
                break
        return obs, total_reward, done, info

    def reset(self):
        return self.env.reset()


class StuckMonitor:
    """Ends the episode when Mario stops progressing (the anti-stuck trick)."""

    def __init__(self, env, stuck_limit=Config.STUCK_LIMIT,
                 inactivity_limit=Config.INACTIVITY_LIMIT,
                 max_steps=Config.MAX_AGENT_STEPS):
        self.env = env
        self.stuck_limit = stuck_limit
        self.inactivity_limit = inactivity_limit
        self.max_steps = max_steps
        self.reset_stats()

    def reset_stats(self):
        self.max_x = 0
        self.stuck_steps = 0
        self.inactive_steps = 0
        self.total_steps = 0
        self.last_score = 0
        self.last_coins = 0
        self.last_time = 400

    def reset(self):
        self.reset_stats()
        return self.env.reset()

    def step(self, action: int):
        obs, reward, done, info = self.env.step(action)
        self.total_steps += 1

        x = info.get('x_pos', 0)
        score = info.get('score', 0)
        coins = info.get('coins', 0)
        time_left = info.get('time', 400)
        life = info.get('life', 2)

        if x > self.max_x:
            self.max_x = x
            self.stuck_steps = 0
        else:
            self.stuck_steps += 1

        active = (score > self.last_score or coins > self.last_coins
                  or time_left < self.last_time)
        self.inactive_steps = 0 if active else self.inactive_steps + 1

        info['stuck_steps'] = self.stuck_steps
        info['inactive_steps'] = self.inactive_steps
        info['progress_x'] = self.max_x

        self.last_score = score
        self.last_coins = coins
        self.last_time = time_left

        # --- anti-stuck termination ---
        if self.stuck_steps >= self.stuck_limit:
            done = True
            info['stuck'] = True
        elif self.inactive_steps >= self.inactivity_limit:
            done = True
            info['inactive'] = True
        elif self.total_steps >= self.max_steps:
            done = True
            info['timeout'] = True
        elif life == 0:
            done = True
            info['died'] = True
            self.reset_stats()
            return obs, reward, done, info

        return obs, reward, done, info


class CurriculumGoal:
    """NeuroMario-style ephemeral goal: reach max_x + offset, then re-target."""

    def __init__(self):
        self.offset = Config.GOAL_MIN_OFFSET
        self.history: List[Tuple[int, bool]] = []
        self.reached = False
        self.baseline = 0

    def reset(self, start_x: int = 0):
        self.history.clear()
        self.reached = False
        self.baseline = start_x

    def target(self) -> int:
        return self.baseline + self.offset

    def update(self, max_x: int) -> bool:
        """Returns True when the current goal was reached."""
        if max_x >= self.target():
            self.reached = True
            self.baseline = max_x
            return True
        return False

    def record(self, max_x: int, success: bool) -> None:
        self.history.append((max_x, success))

    def adjust(self) -> None:
        window = self.history[-Config.GOAL_BUFFER_SIZE:]
        if not window:
            return
        success_rate = sum(1 for _, ok in window if ok) / len(window)
        if success_rate > 0.8 and self.offset < Config.GOAL_MAX_OFFSET:
            self.offset = min(Config.GOAL_MAX_OFFSET, self.offset + 15)
            self.history.clear()
        elif success_rate < 0.2 and self.offset > Config.GOAL_MIN_OFFSET:
            self.offset = max(Config.GOAL_MIN_OFFSET, self.offset - 10)
            self.history.clear()

    @property
    def success_rate(self) -> float:
        if not self.history:
            return 0.0
        return sum(1 for _, ok in self.history if ok) / len(self.history)


class RewardShaper:
    """Shaped reward + discrete event detection.

    Progress, goal, coins and score come from the counters; the discrete events
    (enemy stomp, brick break, mushroom, secret, damage) are inferred from score
    deltas, because ``gym-super-mario-bros`` does not expose them.

    ``frame_skip`` is the *actual* skip in use. The idle penalty has to scale with
    it: one agent decision at frameskip 10 covers ten NES frames, so charging a
    single-step idle penalty under-charges standing still and made idling cheap
    at high frame skip.
    """

    def __init__(self, frame_skip: int = Config.FRAMESKIP):
        self.frame_skip = frame_skip
        self.goal = CurriculumGoal()
        self.prev_x = 0
        self.prev_score = 0
        self.prev_coins = 0
        self.prev_life = 2
        self.prev_time = 400
        self.prev_y = 0
        self.last_block_hit_x = -1000.0
        self.events = np.zeros(6, dtype=np.float32)
        self._ep_reached = False  # la meta se registro este episodio?
        # muertes anteriores (x): persiste entre episodios para pillar el loop
        # de "muere siempre en el mismo sitio". No se limpia en reset().
        from collections import deque as _dq
        self.death_spots = _dq(maxlen=Config.SAME_SPOT_MEMORY)

    def end_episode(self, max_x: int) -> None:
        """Registra UN punto en el curriculum (por episodio, no por step).

        Antes se llamaba record() cada tick: el buffer de 50 se llenaba en
        segundos con falsos y el offset oscilaba sin senal real.
        """
        try:
            self.goal.record(max_x, bool(self._ep_reached))
            self.goal.adjust()
        except Exception:
            pass
        self._ep_reached = False

    def reset(self, start_x: int = 0):
        self.goal.reset(start_x)
        self.prev_x = start_x
        self.prev_score = 0
        self.prev_coins = 0
        self.prev_life = 2
        self.prev_time = 400
        self.prev_y = 0
        self.last_block_hit_x = -1000.0
        self.events = np.zeros(6, dtype=np.float32)

    def _detect_events(self, score_gain: int, coins_gain: int, x: float) -> Dict:
        ev = {'coin': float(coins_gain > 0),
              'powerup': 0.0,
              'enemy': 0.0,
              'block': 0.0,
              'secret': 0.0,
              'damage': 0.0}

        if score_gain >= Config.POWERUP_SCORE_THRESHOLD and coins_gain == 0:
            ev['powerup'] = 1.0
        elif score_gain > 1000 and score_gain % 1000 != 0:
            ev['secret'] = 1.0

        if 50 <= score_gain <= 500 and score_gain not in (200, 1000):
            ev['enemy'] = 1.0

        if abs(x - self.last_block_hit_x) > 5 and 180 <= score_gain <= 220:
            ev['block'] = 1.0
            self.last_block_hit_x = x

        if self.prev_life_flagged:
            ev['damage'] = 1.0

        return ev

    prev_life_flagged = False

    def shape(self, reward: float, info: Dict) -> Tuple[float, Dict]:
        x = info.get('x_pos', 0)
        y = info.get('y_pos', 0)
        score = info.get('score', 0)
        coins = info.get('coins', 0)
        life = info.get('life', 2)
        time_left = info.get('time', 400)

        breakdown = {'progress': 0.0, 'goal': 0.0, 'coin': 0.0,
                     'score': 0.0, 'vertical': 0.0, 'enemy': 0.0,
                     'block': 0.0, 'powerup': 0.0, 'secret': 0.0,
                     # castigos desglosados (antes todo iba en 'penalty')
                     'idle': 0.0, 'backtrack': 0.0, 'damage': 0.0,
                     'death': 0.0, 'samespot': 0.0,
                     'stuck': 0.0, 'inactive': 0.0,
                     'penalty': 0.0}

        dx = x - self.prev_x
        if dx > 0:
            breakdown['progress'] = dx * Config.X_PROGRESS_SCALE
        else:
            # Standing still was locally optimal: NOOP cost nothing while the
            # stuck penalty only fired 100 steps later. Charge every idle step so
            # moving is never the more expensive option.
            breakdown['idle'] = Config.IDLE_PENALTY * self.frame_skip
        if dx < Config.BACKTRACK_THRESHOLD:
            # retroceder mucho: resta proporcional a lo retrocedido
            breakdown['backtrack'] = dx * Config.BACKTRACK_SCALE  # dx<0 -> negativo

        dy = y - self.prev_y
        if dy > 0:
            breakdown['vertical'] = dy * Config.X_PROGRESS_SCALE * 0.5

        coins_gain = coins - self.prev_coins
        score_gain = score - self.prev_score
        self.prev_life_flagged = life < self.prev_life

        ev = self._detect_events(score_gain, coins_gain, x)
        self.events = np.array([ev['coin'], ev['powerup'], ev['enemy'],
                                ev['block'], ev['secret'], ev['damage']],
                               dtype=np.float32)

        if ev['coin']:
            breakdown['coin'] = Config.COIN_REWARD * max(1, coins_gain)
        if ev['enemy']:
            breakdown['enemy'] = Config.ENEMY_REWARD
        if ev['block']:
            breakdown['block'] = Config.BLOCK_REWARD
        if ev['powerup']:
            breakdown['powerup'] = Config.POWERUP_REWARD
        if ev['secret']:
            breakdown['secret'] = Config.SECRET_REWARD
        if ev['damage']:
            # perder powerup / encogerse / dano: castigo directo
            breakdown['damage'] = Config.DAMAGE_PENALTY

        if score_gain > 0:
            breakdown['score'] = min(
                score_gain * Config.SCORE_REWARD_SCALE, Config.MAX_SCORE_REWARD)

        reached = self.goal.update(info.get('progress_x', x))
        if reached:
            breakdown['goal'] = Config.GOAL_REWARD
            self._ep_reached = True

        if info.get('stuck'):
            breakdown['stuck'] = Config.STUCK_PENALTY
        if info.get('inactive'):
            breakdown['inactive'] = Config.STUCK_PENALTY * 0.5
        if life < self.prev_life:
            # perder la vida / morir: base + extra si es el mismo sitio de antes
            breakdown['death'] = Config.DEATH_PENALTY
            if any(abs(x - s) <= Config.SAME_SPOT_RADIUS for s in self.death_spots):
                breakdown['samespot'] = Config.SAME_SPOT_PENALTY
                info['samespot'] = True
            self.death_spots.append(x)

        breakdown['penalty'] = (breakdown['idle'] + breakdown['backtrack']
                                 + breakdown['damage'] + breakdown['death']
                                 + breakdown['samespot'] + breakdown['stuck']
                                 + breakdown['inactive'])

        time_bonus = 0.0
        if info.get('flag_get'):
            time_bonus = 5.0
            self.levels_done = getattr(self, 'levels_done', 0) + 1
        breakdown['progress'] += time_bonus

        total = reward + sum(breakdown.values())

        self.prev_x = x
        self.prev_y = y
        self.prev_score = score
        self.prev_coins = coins
        self.prev_life = life
        self.prev_time = time_left

        info['events'] = self.events
        info['penalties'] = {k: breakdown[k] for k in
                             ('idle', 'backtrack', 'damage', 'death',
                              'samespot', 'stuck', 'inactive')}
        return total, breakdown


class MarioEnvironment:
    """gym-super-mario-bros wrapper exposing (screen, semantic) + NeuroMario shaping."""

    def __init__(self, env_name: str = 'SuperMarioBros-v0',
                 frame_skip: int = Config.FRAMESKIP,
                 stuck_limit: int = Config.STUCK_LIMIT,
                 inactivity_limit: int = Config.INACTIVITY_LIMIT,
                 max_steps: int = Config.MAX_AGENT_STEPS,
                 stage: str = '1-1'):
        if not GYM_AVAILABLE:
            raise ImportError('gym-super-mario-bros not installed')

        world, level = stage.split('-')
        target = (int(world), int(level))

        try:
            # gym-super-mario-bros >= 9
            self.raw_env = gym_super_mario_bros.make(env_name, target=target)
        except TypeError:
            # gym-super-mario-bros < 9
            self.raw_env = gym_super_mario_bros.make(env_name, stages=[stage])

        self.env = JoypadSpace(self.raw_env, SIMPLE_MOVEMENT)
        self.frameskip = FrameSkip(self.env, frame_skip)
        self.stuck = StuckMonitor(self.frameskip, stuck_limit,
                                  inactivity_limit, max_steps)
        self.shaper = RewardShaper(frame_skip=frame_skip)

        self.n_actions = Config.N_ACTIONS
        self.action_names = Config.ACTION_NAMES
        self.prev_action = 0
        self.deaths = 0
        self.levels_completed = 0
        # frame stack x4: velocidad visible (ir vs venir). deque de [3,H,W].
        from collections import deque as _dq
        self._frames: _dq = _dq(maxlen=Config.FRAME_STACK)

    def _stacked(self) -> np.ndarray:
        """Apila deque -> [12,240,256]. Rellena repitiendo el mas viejo."""
        import numpy as _np
        frames = list(self._frames)
        while len(frames) < Config.FRAME_STACK:
            frames = [frames[0]] + frames
        return _np.concatenate(frames, axis=0).astype(_np.float32)

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        result = self.stuck.reset()
        obs = result[0] if isinstance(result, tuple) else result

        info = {}
        frame = self._screen(obs)
        self._frames.clear()
        for _ in range(Config.FRAME_STACK):
            self._frames.append(frame)
        screen = self._stacked()
        semantic, _ = self._semantic(info, 0.0, 0.0)
        self.shaper.reset(0)
        self.prev_action = 0
        return screen, semantic

    def step(self, action: int) -> Tuple[np.ndarray, np.ndarray, float, bool, Dict]:
        obs, raw_reward, done, info = self.stuck.step(action)

        # gym fires done=True on death while info['life'] still lags at its
        # previous value, so classify the termination ourselves.
        if done:
            if info.get('stuck') or info.get('inactive') or info.get('timeout'):
                pass
            elif info.get('flag_get'):
                info['level_complete'] = True
                self.levels_completed += 1
            else:
                info['died'] = True
                self.deaths += 1

        self.prev_action = action
        reward, breakdown = self.shaper.shape(raw_reward, info)
        info['reward_breakdown'] = breakdown
        # Senal de causa exacta: vector + texto para el log del momento justo.
        try:
            info['cause_vec'] = breakdown_to_cause_vec(breakdown)
            info['cause_text'] = breakdown_to_text(breakdown)
            info['cause_has'] = bool(info['cause_vec'].sum() > 0)
        except Exception:
            pass
        info['goal_target'] = self.shaper.goal.target()
        info['goal_offset'] = self.shaper.goal.offset
        info['goal_success_rate'] = self.shaper.goal.success_rate
        if done:
            # curriculum por episodio: un solo punto con el max_x final
            try:
                self.shaper.end_episode(info.get('progress_x', 0))
            except Exception:
                pass

        self._frames.append(self._screen(obs))
        screen = self._stacked()
        semantic, _ = self._semantic(info, self.prev_x, self.prev_y)
        return screen, semantic, reward, done, info

    prev_x = 0
    prev_y = 0

    def _screen(self, obs: np.ndarray) -> np.ndarray:
        arr = np.asarray(obs)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.ndim == 3 and arr.shape[-1] != 3:
            arr = np.stack([arr[..., 0]] * 3, axis=-1)
        if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
            arr = np.transpose(arr, (1, 2, 0))
        arr = arr.astype(np.float32)
        return np.transpose(arr, (2, 0, 1)) / 255.0

    def _semantic(self, info: Dict, prev_x: float, prev_y: float) -> Tuple[np.ndarray, int]:
        """Build the semantic vector from the env info dict (the real signal)."""
        x = float(info.get('x_pos', 0) or 0)
        y = float(info.get('y_pos', 0) or 0)
        score = float(info.get('score', 0) or 0)
        coins = float(info.get('coins', 0) or 0)
        life = float(info.get('life', 2) or 0)
        time_left = float(info.get('time', 400) or 0)
        stage = int(info.get('stage', 1) or 1)
        world = int(info.get('world', 1) or 1)
        flag_get = float(info.get('flag_get', False))

        vx = np.clip((x - prev_x) / 16.0, -2.0, 2.0)
        vy = np.clip((y - prev_y) / 16.0, -2.0, 2.0)
        goal = float(self.shaper.goal.target())
        # BUG viejo: goal_dist ya venia /200 y abajo se dividia otra vez
        # (/200.0) -> la meta era ~0 siempre, invisible para la red.
        goal_dist = float(np.clip((goal - x) / 200.0, -1.0, 2.0))

        sem = np.array([
            x / 256.0,
            y / 240.0,
            score / 10000.0,
            coins / 100.0,
            life / 3.0,
            time_left / 400.0,
            vx,
            vy,
            goal_dist,
            goal / 4000.0,
            float(stage % 4) / 4.0,
            float(world % 4) / 4.0,
            float(info.get('stuck_steps', 0)) / Config.STUCK_LIMIT,
            float(info.get('inactive_steps', 0)) / Config.INACTIVITY_LIMIT,
            flag_get,
            float(self.prev_action == 0),
            float(self.prev_action in (1, 2, 3, 4)),
            float(self.prev_action == 4),
            float(self.prev_action == 5),
            float(self.prev_action == 6),
            1.0 if info.get('stuck') else 0.0,
            1.0 if info.get('died') else 0.0,
            float(self.shaper.goal.success_rate),
            float(info.get('progress_x', x)) / 4000.0,
            float(np.sin(x / 64.0)),
            float(np.cos(y / 64.0)),
            float(x - self.shaper.prev_x) / 16.0,
            # discrete game events, so the network can *see* them happen
            float(self.shaper.events[0]),   # coin collected
            float(self.shaper.events[1]),   # mushroom / fire flower
            float(self.shaper.events[2]),   # enemy stomped
            float(self.shaper.events[3]),   # brick broken
            float(self.shaper.events[4]),   # secret exit
            float(self.shaper.events[5]),   # took damage
        ], dtype=np.float32)

        sem = np.nan_to_num(sem, nan=0.0, posinf=1.0, neginf=-1.0)
        self.prev_x, self.prev_y = x, y
        return sem, int(x)

    def close(self):
        self.raw_env.close()


class ExplorationSchedule:
    """NeuroMario epsilon decay, with a hard anti-stuck override."""

    def __init__(self, start: float = 1.0, floor: float = 0.05,
                 decay: float = Config.EXPLORATION_DECAY_RATE):
        self.epsilon = start
        self.start = start
        self.floor = floor
        self.decay = decay

    def step(self) -> None:
        self.epsilon = max(self.floor, self.epsilon * self.decay)

    def reset(self) -> None:
        self.epsilon = self.start


class VecMarioEnv:
    """N Mario environments stepped in lockstep, one batched policy.

    The NES emulator is single-threaded per env and the GPU sat at ~14% during
    training, so running several envs against one batched forward pass is far
    cheaper than several processes. Each env keeps its own reward shaper, stuck
    monitor and curriculum goal.
    """

    def __init__(self, n: int = 4, stage: str = '1-1', **kwargs):
        self.envs = [create_environment(stage=stage, **kwargs)
                     for _ in range(n)]
        self.n = n
        self.action_names = self.envs[0].action_names
        self.n_actions = self.envs[0].n_actions
        self.deaths = sum(e.deaths for e in self.envs)
        self.levels_completed = sum(e.levels_completed for e in self.envs)

    @property
    def goal_offset(self) -> int:
        return int(np.mean([e.shaper.goal.offset for e in self.envs]))

    @property
    def goal_success_rate(self) -> float:
        return float(np.mean([e.shaper.goal.success_rate for e in self.envs]))

    def reset(self):
        out = [e.reset() for e in self.envs]
        self._sync_counters()
        return (np.stack([o[0] for o in out]),
                np.stack([o[1] for o in out]))

    def step(self, actions):
        """Step every env. Envs that report done are auto-reset so the caller
        can keep feeding actions without special-casing per-env termination."""
        results = []
        for e, a in zip(self.envs, actions):
            screen, semantic, reward, done, info = e.step(int(a))
            if done:
                screen, semantic = e.reset()
                info['auto_reset'] = True
            results.append((screen, semantic, reward, done, info))
        self._sync_counters()
        return (np.stack([r[0] for r in results]),
                np.stack([r[1] for r in results]),
                np.array([r[2] for r in results], dtype=np.float32),
                np.array([r[3] for r in results], dtype=bool),
                [r[4] for r in results])

    def _sync_counters(self):
        self.deaths = sum(e.deaths for e in self.envs)
        self.levels_completed = sum(e.levels_completed for e in self.envs)

    def close(self):
        for e in self.envs:
            e.close()


def create_vector_env(n: int = 4, stage: str = '1-1', **kwargs) -> VecMarioEnv:
    """Create ``n`` independent Mario environments for batched rollouts."""
    return VecMarioEnv(n=n, stage=stage, **kwargs)


def create_environment(env_name: str = 'SuperMarioBros-v0', **kwargs) -> MarioEnvironment:
    return MarioEnvironment(env_name, **kwargs)
