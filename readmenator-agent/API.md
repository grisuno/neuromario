# API

## app.py

### compute_real_saliency (function) `def compute_real_saliency(agent, state, aux_features)`
- Defined: `app.py:55`
- Depends on: `trimario4.py`

### run_and_log_activations (function) `def run_and_log_activations(agent, env, num_steps)`
- Defined: `app.py:130`
- Depends on: `trimario4.py`

### plot_activation_3d_and_bars (function) `def plot_activation_3d_and_bars(log_data)`
- Defined: `app.py:215`
- Depends on: `trimario4.py`

### visualize_attention_layers (function) `def visualize_attention_layers(agent, state, aux_features)`
- Defined: `app.py:323`
- Doc: Visualiza mapas de atención en diferentes etapas del procesamiento
- Depends on: `trimario4.py`

## trimario4.py

### _safe_text (function) `def _safe_text(self, x, y, s)`
- Defined: `trimario4.py:34`
- Imported by: `app.py`

### timeout_handler (method) `def timeout_handler(signum, frame)`
- Defined: `trimario4.py:48`
- Imported by: `app.py`

### safe_step_with_timeout (method) `def safe_step_with_timeout(env, action, timeout_seconds)`
- Defined: `trimario4.py:51`
- Doc: Ejecuta step con timeout para detectar bloqueos
- Imported by: `app.py`

### log_detailed_metrics (method) `def log_detailed_metrics(agent, ep, scores, losses, accuracies)`
- Defined: `trimario4.py:181`
- Imported by: `app.py`

### preprocess_frame (method) `def preprocess_frame(frame)`
- Defined: `trimario4.py:552`
- Doc: Versión ultra-robusta: nunca devuelve NaN/inf.
- Imported by: `app.py`

### stack_frames (method) `def stack_frames(stacked, frame, is_new)`
- Defined: `trimario4.py:580`
- Imported by: `app.py`

### compute_focus_quality_score (method) `def compute_focus_quality_score(saliency_map)`
- Defined: `trimario4.py:1797`
- Doc: Calcula score de calidad del foco visual
- Imported by: `app.py`

### get_aux_features (method) `def get_aux_features(info, history)`
- Defined: `trimario4.py:1830`
- Imported by: `app.py`

### __init__ (method) `def __init__(self, device)`
- Defined: `trimario4.py:237`
- Imported by: `app.py`

### extract_event_features (method) `def extract_event_features(self, info)`
- Defined: `trimario4.py:309`
- Imported by: `app.py`

### forward (method) `def forward(self, info, visual_context)`
- Defined: `trimario4.py:365`
- Imported by: `app.py`

### reset (method) `def reset(self)`
- Defined: `trimario4.py:429`
- Imported by: `app.py`

### __init__ (method) `def __init__(self, env, skip)`
- Defined: `trimario4.py:442`
- Imported by: `app.py`

### step (method) `def step(self, action)`
- Defined: `trimario4.py:446`
- Imported by: `app.py`

### __init__ (method) `def __init__(self, env, stuck_limit, inactivity_limit)`
- Defined: `trimario4.py:474`
- Imported by: `app.py`

### reset_stats (method) `def reset_stats(self)`
- Defined: `trimario4.py:480`
- Imported by: `app.py`

### step (method) `def step(self, action)`
- Defined: `trimario4.py:490`
- Imported by: `app.py`

### reset (method) `def reset(self)`
- Defined: `trimario4.py:529`
- Imported by: `app.py`

### __init__ (method) `def __init__(self, device)`
- Defined: `trimario4.py:601`
- Imported by: `app.py`

### forward (method) `def forward(self, x)`
- Defined: `trimario4.py:658`
- Imported by: `app.py`

### __init__ (method) `def __init__(self, in_dim, out_dim, device)`
- Defined: `trimario4.py:687`
- Imported by: `app.py`

### forward (method) `def forward(self, x)`
- Defined: `trimario4.py:715`
- Imported by: `app.py`

### compute_plasticity_gradient (method) `def compute_plasticity_gradient(self, x, output, td_error)`
- Defined: `trimario4.py:755`
- Imported by: `app.py`

### post_step_update (method) `def post_step_update(self)`
- Defined: `trimario4.py:806`
- Imported by: `app.py`

### __init__ (method) `def __init__(self, input_dim, output_dim, aux_dim, device)`
- Defined: `trimario4.py:816`
- Imported by: `app.py`

### forward (method) `def forward(self, stacked_frame, aux_features, goal_x, current_x, info, visualization_mode)`
- Defined: `trimario4.py:841`
- Imported by: `app.py`

### forward_legacy (method) `def forward_legacy(self, stacked_frame, aux_features, goal_x, current_x)`
- Defined: `trimario4.py:913`
- Doc: Wrapper para compatibilidad con código que espera 5 retornos
- Imported by: `app.py`

### __init__ (method) `def __init__(self, dim)`
- Defined: `trimario4.py:925`
- Imported by: `app.py`

### forward (method) `def forward(self, visual_features, audio_features, semantic_features, td_error)`
- Defined: `trimario4.py:961`
- Imported by: `app.py`

### reset_fatigue (method) `def reset_fatigue(self)`
- Defined: `trimario4.py:1136`
- Imported by: `app.py`

### __init__ (method) `def __init__(self, capacity)`
- Defined: `trimario4.py:1143`
- Imported by: `app.py`

### __len__ (method) `def __len__(self)`
- Defined: `trimario4.py:1149`
- Imported by: `app.py`

### add (method) `def add(self, experience, td_error)`
- Defined: `trimario4.py:1152`
- Imported by: `app.py`

### sample (method) `def sample(self, batch_size, beta)`
- Defined: `trimario4.py:1163`
- Imported by: `app.py`

### update_priorities (method) `def update_priorities(self, indices, td_errors)`
- Defined: `trimario4.py:1200`
- Imported by: `app.py`

### __init__ (method) `def __init__(self, n_actions, input_dim, hidden_dim)`
- Defined: `trimario4.py:1207`
- Imported by: `app.py`

### forward (method) `def forward(self, x)`
- Defined: `trimario4.py:1248`
- Imported by: `app.py`

### forward_with_cache (method) `def forward_with_cache(self, x)`
- Defined: `trimario4.py:1271`
- Imported by: `app.py`

### reset_memory (method) `def reset_memory(self)`
- Defined: `trimario4.py:1291`
- Doc: Reinicia el buffer de memoria episódica del hemisferio izquierdo
- Imported by: `app.py`

### store_experience (method) `def store_experience(self, x)`
- Defined: `trimario4.py:1300`
- Doc: Almacena una experiencia en el buffer circular
- Imported by: `app.py`

### __init__ (method) `def __init__(self, device)`
- Defined: `trimario4.py:1324`
- Imported by: `app.py`

### store_episode (method) `def store_episode(self, event_type, liquid_state)`
- Defined: `trimario4.py:1355`
- Imported by: `app.py`

### retrieve_similar_episode (method) `def retrieve_similar_episode(self, current_state)`
- Defined: `trimario4.py:1400`
- Imported by: `app.py`

### __init__ (method) `def __init__(self, n_actions, device)`
- Defined: `trimario4.py:1451`
- Imported by: `app.py`

### act (method) `def act(self, state, aux_features, epsilon, info)`
- Defined: `trimario4.py:1511`
- Imported by: `app.py`

### remember (method) `def remember(self, s, a, r, s_next, aux_s, aux_s_next, done, td_error)`
- Defined: `trimario4.py:1542`
- Imported by: `app.py`

### update_target_networks (method) `def update_target_networks(self, tau)`
- Defined: `trimario4.py:1554`
- Imported by: `app.py`

### replay (method) `def replay(self, batch_size, gamma)`
- Defined: `trimario4.py:1566`
- Imported by: `app.py`

### propose_goal (method) `def propose_goal(self, current_x, episode_num)`
- Defined: `trimario4.py:1746`
- Imported by: `app.py`

### is_goal_achieved (method) `def is_goal_achieved(self, goal_x, current_x)`
- Defined: `trimario4.py:1770`
- Imported by: `app.py`

### reset (method) `def reset(self)`
- Defined: `trimario4.py:1773`
- Imported by: `app.py`
