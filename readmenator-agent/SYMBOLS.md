# Symbols

| Symbol | Kind | File:Line | Signature |
|--------|------|-----------|-----------|
| `compute_real_saliency` | function | `app.py:55` | `def compute_real_saliency(agent, state, aux_features)` |
| `plot_activation_3d_and_bars` | function | `app.py:215` | `def plot_activation_3d_and_bars(log_data)` |
| `run_and_log_activations` | function | `app.py:130` | `def run_and_log_activations(agent, env, num_steps)` |
| `visualize_attention_layers` | function | `app.py:323` | `def visualize_attention_layers(agent, state, aux_features)` |
| `AudioFeatureGenerator` | class | `trimario4.py:236` | `class AudioFeatureGenerator(Module)` |
| `Config` | class | `trimario4.py:68` | `class Config` |
| `CorpusCallosum` | class | `trimario4.py:924` | `class CorpusCallosum(Module)` |
| `EpisodicMemory` | class | `trimario4.py:1323` | `class EpisodicMemory(Module)` |
| `FrameSkip` | class | `trimario4.py:441` | `class FrameSkip(Wrapper)` |
| `LeftHemisphere` | class | `trimario4.py:1206` | `class LeftHemisphere(Module)` |
| `PrioritizedReplayBuffer` | class | `trimario4.py:1142` | `class PrioritizedReplayBuffer` |
| `RightHemisphere` | class | `trimario4.py:815` | `class RightHemisphere(Module)` |
| `StableLiquidNeuron` | class | `trimario4.py:686` | `class StableLiquidNeuron(Module)` |
| `StuckMonitor` | class | `trimario4.py:473` | `class StuckMonitor(Wrapper)` |
| `TimeoutException` | class | `trimario4.py:45` | `class TimeoutException(Exception)` |
| `TricameralMarioAgent` | class | `trimario4.py:1450` | `class TricameralMarioAgent(Module)` |
| `VisualFeatureExtractor` | class | `trimario4.py:600` | `class VisualFeatureExtractor(Module)` |
| `__init__` | method | `trimario4.py:237` | `def __init__(self, device)` |
| `__init__` | method | `trimario4.py:442` | `def __init__(self, env, skip)` |
| `__init__` | method | `trimario4.py:474` | `def __init__(self, env, stuck_limit, inactivity_limit)` |
| `__init__` | method | `trimario4.py:601` | `def __init__(self, device)` |
| `__init__` | method | `trimario4.py:687` | `def __init__(self, in_dim, out_dim, device)` |
| `__init__` | method | `trimario4.py:816` | `def __init__(self, input_dim, output_dim, aux_dim, device)` |
| `__init__` | method | `trimario4.py:925` | `def __init__(self, dim)` |
| `__init__` | method | `trimario4.py:1143` | `def __init__(self, capacity)` |
| `__init__` | method | `trimario4.py:1207` | `def __init__(self, n_actions, input_dim, hidden_dim)` |
| `__init__` | method | `trimario4.py:1324` | `def __init__(self, device)` |
| `__init__` | method | `trimario4.py:1451` | `def __init__(self, n_actions, device)` |
| `__len__` | method | `trimario4.py:1149` | `def __len__(self)` |
| `_safe_text` | function | `trimario4.py:34` | `def _safe_text(self, x, y, s)` |
| `act` | method | `trimario4.py:1511` | `def act(self, state, aux_features, epsilon, info)` |
| `add` | method | `trimario4.py:1152` | `def add(self, experience, td_error)` |
| `compute_focus_quality_score` | method | `trimario4.py:1797` | `def compute_focus_quality_score(saliency_map)` |
| `compute_plasticity_gradient` | method | `trimario4.py:755` | `def compute_plasticity_gradient(self, x, output, td_error)` |
| `extract_event_features` | method | `trimario4.py:309` | `def extract_event_features(self, info)` |
| `forward` | method | `trimario4.py:365` | `def forward(self, info, visual_context)` |
| `forward` | method | `trimario4.py:658` | `def forward(self, x)` |
| `forward` | method | `trimario4.py:715` | `def forward(self, x)` |
| `forward` | method | `trimario4.py:841` | `def forward(self, stacked_frame, aux_features, goal_x, current_x, info, visualization_mode)` |
| `forward` | method | `trimario4.py:961` | `def forward(self, visual_features, audio_features, semantic_features, td_error)` |
| `forward` | method | `trimario4.py:1248` | `def forward(self, x)` |
| `forward_legacy` | method | `trimario4.py:913` | `def forward_legacy(self, stacked_frame, aux_features, goal_x, current_x)` |
| `forward_with_cache` | method | `trimario4.py:1271` | `def forward_with_cache(self, x)` |
| `get_aux_features` | method | `trimario4.py:1830` | `def get_aux_features(info, history)` |
| `is_goal_achieved` | method | `trimario4.py:1770` | `def is_goal_achieved(self, goal_x, current_x)` |
| `log_detailed_metrics` | method | `trimario4.py:181` | `def log_detailed_metrics(agent, ep, scores, losses, accuracies)` |
| `post_step_update` | method | `trimario4.py:806` | `def post_step_update(self)` |
| `preprocess_frame` | method | `trimario4.py:552` | `def preprocess_frame(frame)` |
| `propose_goal` | method | `trimario4.py:1746` | `def propose_goal(self, current_x, episode_num)` |
| `remember` | method | `trimario4.py:1542` | `def remember(self, s, a, r, s_next, aux_s, aux_s_next, done, td_error)` |
| `replay` | method | `trimario4.py:1566` | `def replay(self, batch_size, gamma)` |
| `reset` | method | `trimario4.py:429` | `def reset(self)` |
| `reset` | method | `trimario4.py:529` | `def reset(self)` |
| `reset` | method | `trimario4.py:1773` | `def reset(self)` |
| `reset_fatigue` | method | `trimario4.py:1136` | `def reset_fatigue(self)` |
| `reset_memory` | method | `trimario4.py:1291` | `def reset_memory(self)` |
| `reset_stats` | method | `trimario4.py:480` | `def reset_stats(self)` |
| `retrieve_similar_episode` | method | `trimario4.py:1400` | `def retrieve_similar_episode(self, current_state)` |
| `safe_step_with_timeout` | method | `trimario4.py:51` | `def safe_step_with_timeout(env, action, timeout_seconds)` |
| `sample` | method | `trimario4.py:1163` | `def sample(self, batch_size, beta)` |
| `stack_frames` | method | `trimario4.py:580` | `def stack_frames(stacked, frame, is_new)` |
| `step` | method | `trimario4.py:446` | `def step(self, action)` |
| `step` | method | `trimario4.py:490` | `def step(self, action)` |
| `store_episode` | method | `trimario4.py:1355` | `def store_episode(self, event_type, liquid_state)` |
| `store_experience` | method | `trimario4.py:1300` | `def store_experience(self, x)` |
| `timeout_handler` | method | `trimario4.py:48` | `def timeout_handler(signum, frame)` |
| `update_priorities` | method | `trimario4.py:1200` | `def update_priorities(self, indices, td_errors)` |
| `update_target_networks` | method | `trimario4.py:1554` | `def update_target_networks(self, tau)` |
