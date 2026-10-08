# root

*Community 0 | 2 files | cohesion 1.00*

## Definition

This community groups 2 file(s) rooted at `root` with dominant language py (cohesion 1.00). Central symbols: `AudioFeatureGenerator`, `Config`, `CorpusCallosum`, `EpisodicMemory`, `FrameSkip`, `LeftHemisphere`, `PrioritizedReplayBuffer`, `RightHemisphere`. Core file: `trimario4.py` (64 symbols). Documented purpose: Autor: Gris Iscomeback Correo electrónico: grisiscomeback[at]gmail[dot]com Fecha de creación: xx/xx/xxxx Licencia: GPL v3  Descripción:.

## Files

| File | Language | Layer | Symbols | Doc |
|------|----------|-------|---------|-----|
| `app.py` | py | utility | 4 | yes |
| `trimario4.py` | py | utility | 64 | no |

## Key Symbols

- `compute_real_saliency` (function, `app.py:55`) `def compute_real_saliency(agent, state, aux_features)`
- `run_and_log_activations` (function, `app.py:130`) `def run_and_log_activations(agent, env, num_steps)`
- `plot_activation_3d_and_bars` (function, `app.py:215`) `def plot_activation_3d_and_bars(log_data)`
- `visualize_attention_layers` (function, `app.py:323`) `def visualize_attention_layers(agent, state, aux_features)` - Visualiza mapas de atención en diferentes etapas del procesamiento
- `_safe_text` (function, `trimario4.py:34`) `def _safe_text(self, x, y, s)`
- `TimeoutException` (class, `trimario4.py:45`) `class TimeoutException(Exception)`
- `timeout_handler` (method, `trimario4.py:48`) `def timeout_handler(signum, frame)`
- `safe_step_with_timeout` (method, `trimario4.py:51`) `def safe_step_with_timeout(env, action, timeout_seconds)` - Ejecuta step con timeout para detectar bloqueos
- `Config` (class, `trimario4.py:68`) `class Config`
- `log_detailed_metrics` (method, `trimario4.py:181`) `def log_detailed_metrics(agent, ep, scores, losses, accuracies)`
- `AudioFeatureGenerator` (class, `trimario4.py:236`) `class AudioFeatureGenerator(Module)`
- `__init__` (method, `trimario4.py:237`) `def __init__(self, device)`
- `extract_event_features` (method, `trimario4.py:309`) `def extract_event_features(self, info)`
- `forward` (method, `trimario4.py:365`) `def forward(self, info, visual_context)`
- `reset` (method, `trimario4.py:429`) `def reset(self)`
- `FrameSkip` (class, `trimario4.py:441`) `class FrameSkip(Wrapper)`
- `__init__` (method, `trimario4.py:442`) `def __init__(self, env, skip)`
- `step` (method, `trimario4.py:446`) `def step(self, action)`
- `StuckMonitor` (class, `trimario4.py:473`) `class StuckMonitor(Wrapper)`
- `__init__` (method, `trimario4.py:474`) `def __init__(self, env, stuck_limit, inactivity_limit)`
- `reset_stats` (method, `trimario4.py:480`) `def reset_stats(self)`
- `step` (method, `trimario4.py:490`) `def step(self, action)`
- `reset` (method, `trimario4.py:529`) `def reset(self)`
- `preprocess_frame` (method, `trimario4.py:552`) `def preprocess_frame(frame)` - Versión ultra-robusta: nunca devuelve NaN/inf.
- `stack_frames` (method, `trimario4.py:580`) `def stack_frames(stacked, frame, is_new)`
- `VisualFeatureExtractor` (class, `trimario4.py:600`) `class VisualFeatureExtractor(Module)`
- `__init__` (method, `trimario4.py:601`) `def __init__(self, device)`
- `forward` (method, `trimario4.py:658`) `def forward(self, x)`
- `StableLiquidNeuron` (class, `trimario4.py:686`) `class StableLiquidNeuron(Module)`
- `__init__` (method, `trimario4.py:687`) `def __init__(self, in_dim, out_dim, device)`

## Internal vs External Edges

- Internal resolved imports (EXTRACTED): 1
- Cross-boundary resolved imports (EXTRACTED): 0

## Connections

- [INFERRED] shares_context community 0 <-> 1 (strength 0.5): Inferred shared context (layer utility) with no import path between community 0 (root) and community 1 (orphans).

## Risks

- No scoped security, taint, cycle, or layer risks.

## Open Questions

- Why do 1 file(s) lack file-level docs (e.g. `trimario4.py`)? What purpose do they serve?
- What would break if the most connected file in root changed?
- Should root be split, given cohesion 1.00?

## Sources

- `app.py`
- `trimario4.py`
