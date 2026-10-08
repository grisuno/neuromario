# API

## app.py
Depends on: `trimario4.py`
- `compute_real_saliency` (function) `app.py:55` `def compute_real_saliency(agent, state, aux_features)`
- `run_and_log_activations` (function) `app.py:130` `def run_and_log_activations(agent, env, num_steps)`
- `plot_activation_3d_and_bars` (function) `app.py:215` `def plot_activation_3d_and_bars(log_data)`
- `visualize_attention_layers` (function) `app.py:323` `def visualize_attention_layers(agent, state, aux_features)` -- Visualiza mapas de atención en diferentes etapas del procesamiento

## trimario4.py
Imported by: `app.py`
- `TimeoutException.timeout_handler` (method) `trimario4.py:48` `def timeout_handler(signum, frame)`
- `TimeoutException.safe_step_with_timeout` (method) `trimario4.py:51` `def safe_step_with_timeout(env, action, timeout_seconds)` -- Ejecuta step con timeout para detectar bloqueos
- `Config.log_detailed_metrics` (method) `trimario4.py:181` `def log_detailed_metrics(agent, ep, scores, losses, accuracies)`
- `AudioFeatureGenerator.__init__` (method) `trimario4.py:237` `def __init__(self, device)`
- `AudioFeatureGenerator.extract_event_features` (method) `trimario4.py:309` `def extract_event_features(self, info)`
- `AudioFeatureGenerator.forward` (method) `trimario4.py:365` `def forward(self, info, visual_context)`
- `AudioFeatureGenerator.reset` (method) `trimario4.py:429` `def reset(self)`
- `FrameSkip.__init__` (method) `trimario4.py:442` `def __init__(self, env, skip)`
- `FrameSkip.step` (method) `trimario4.py:446` `def step(self, action)`
- `StuckMonitor.__init__` (method) `trimario4.py:474` `def __init__(self, env, stuck_limit, inactivity_limit)`
- `StuckMonitor.reset_stats` (method) `trimario4.py:480` `def reset_stats(self)`
- `StuckMonitor.step` (method) `trimario4.py:490` `def step(self, action)`
- `StuckMonitor.reset` (method) `trimario4.py:529` `def reset(self)`
- `StuckMonitor.preprocess_frame` (method) `trimario4.py:552` `def preprocess_frame(frame)` -- Versión ultra-robusta: nunca devuelve NaN/inf.
- `StuckMonitor.stack_frames` (method) `trimario4.py:580` `def stack_frames(stacked, frame, is_new)`
- `VisualFeatureExtractor.__init__` (method) `trimario4.py:601` `def __init__(self, device)`
- `VisualFeatureExtractor.forward` (method) `trimario4.py:658` `def forward(self, x)`
- `StableLiquidNeuron.__init__` (method) `trimario4.py:687` `def __init__(self, in_dim, out_dim, device)`
- `StableLiquidNeuron.forward` (method) `trimario4.py:715` `def forward(self, x)`
- `StableLiquidNeuron.compute_plasticity_gradient` (method) `trimario4.py:755` `def compute_plasticity_gradient(self, x, output, td_error)`
- `StableLiquidNeuron.post_step_update` (method) `trimario4.py:806` `def post_step_update(self)`
- `RightHemisphere.__init__` (method) `trimario4.py:816` `def __init__(self, input_dim, output_dim, aux_dim, device)`
- `RightHemisphere.forward` (method) `trimario4.py:841` `def forward(self, stacked_frame, aux_features, goal_x, current_x, info, visualization_mode)`
- `RightHemisphere.forward_legacy` (method) `trimario4.py:913` `def forward_legacy(self, stacked_frame, aux_features, goal_x, current_x)` -- Wrapper para compatibilidad con código que espera 5 retornos
- `CorpusCallosum.__init__` (method) `trimario4.py:925` `def __init__(self, dim)`
- `CorpusCallosum.forward` (method) `trimario4.py:961` `def forward(self, visual_features, audio_features, semantic_features, td_error)`
- `CorpusCallosum.reset_fatigue` (method) `trimario4.py:1136` `def reset_fatigue(self)`
- `PrioritizedReplayBuffer.__init__` (method) `trimario4.py:1143` `def __init__(self, capacity)`
- `PrioritizedReplayBuffer.add` (method) `trimario4.py:1152` `def add(self, experience, td_error)`
- `PrioritizedReplayBuffer.sample` (method) `trimario4.py:1163` `def sample(self, batch_size, beta)`
- `PrioritizedReplayBuffer.update_priorities` (method) `trimario4.py:1200` `def update_priorities(self, indices, td_errors)`
- `LeftHemisphere.__init__` (method) `trimario4.py:1207` `def __init__(self, n_actions, input_dim, hidden_dim)`
- `LeftHemisphere.forward` (method) `trimario4.py:1248` `def forward(self, x)`
- `LeftHemisphere.forward_with_cache` (method) `trimario4.py:1271` `def forward_with_cache(self, x)`
- `LeftHemisphere.reset_memory` (method) `trimario4.py:1291` `def reset_memory(self)` -- Reinicia el buffer de memoria episódica del hemisferio izquierdo
- `LeftHemisphere.store_experience` (method) `trimario4.py:1300` `def store_experience(self, x)` -- Almacena una experiencia en el buffer circular
- `EpisodicMemory.__init__` (method) `trimario4.py:1324` `def __init__(self, device)`
- `EpisodicMemory.store_episode` (method) `trimario4.py:1355` `def store_episode(self, event_type, liquid_state)`
- `EpisodicMemory.retrieve_similar_episode` (method) `trimario4.py:1400` `def retrieve_similar_episode(self, current_state)`
- `TricameralMarioAgent.__init__` (method) `trimario4.py:1451` `def __init__(self, n_actions, device)`
- `TricameralMarioAgent.act` (method) `trimario4.py:1511` `def act(self, state, aux_features, epsilon, info)`
- `TricameralMarioAgent.remember` (method) `trimario4.py:1542` `def remember(self, s, a, r, s_next, aux_s, aux_s_next, done, td_error)`
- `TricameralMarioAgent.update_target_networks` (method) `trimario4.py:1554` `def update_target_networks(self, tau)`
- `TricameralMarioAgent.replay` (method) `trimario4.py:1566` `def replay(self, batch_size, gamma)`
- `TricameralMarioAgent.propose_goal` (method) `trimario4.py:1746` `def propose_goal(self, current_x, episode_num)`
- `TricameralMarioAgent.is_goal_achieved` (method) `trimario4.py:1770` `def is_goal_achieved(self, goal_x, current_x)`
- `TricameralMarioAgent.reset` (method) `trimario4.py:1773` `def reset(self)`
- `TricameralMarioAgent.compute_focus_quality_score` (method) `trimario4.py:1797` `def compute_focus_quality_score(saliency_map)` -- Calcula score de calidad del foco visual
- `TricameralMarioAgent.get_aux_features` (method) `trimario4.py:1830` `def get_aux_features(info, history)`
