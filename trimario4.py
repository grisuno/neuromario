import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import cv2
import numpy as np
import random
from collections import deque
import pickle
import os
from tqdm import tqdm
import matplotlib

try:
    matplotlib.use('TkAgg')
except ImportError:
    pass
import matplotlib.pyplot as plt

import time
import signal



_orig_text = plt.Axes.text
def _safe_text(self, x, y, s, *a, **k):
    if not (np.isfinite(x) and np.isfinite(y)):
        print(f'[TEXT-NaN] x={x}  y={y}  s={s!r}')
        x, y = 0.0, 0.0
    if isinstance(s, str) and ('nan' in s.lower() or 'inf' in s.lower()):
        print(f'[TEXT-NaN] string contains nan/inf: {s!r}')
        s = 'NaN-blocked'
    return _orig_text(self, x, y, s, *a, **k)
plt.Axes.text = _safe_text


class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Step timeout - posible deadlock")

def safe_step_with_timeout(env, action, timeout_seconds=5):
    """Ejecuta step con timeout para detectar bloqueos"""
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        result = env.step(action)
        signal.alarm(0)
        return result
    except TimeoutException:
        print("\n⚠️ TIMEOUT DETECTADO - Reiniciando episodio...")
        signal.alarm(0)
        return None, 0, True, {'timeout': True}

# ==============================================================================
# CONFIG
# ==============================================================================
class Config:
    DEVICE = "cpu"
    STUCK_LIMIT = 100
    INACTIVITY_LIMIT = 150
    FRAMESKIP = 8
    MAX_AGENT_STEPS = 20 * STUCK_LIMIT
    MIN_MEMORY_BEFORE_LEARN = 2 * STUCK_LIMIT
    GOAL_MIN_OFFSET = 30
    GOAL_MAX_OFFSET = 120
    GOAL_BUFFER_SIZE = 50
    POWER_UP_THRESHOLD = 2000
    DEATH_PENALTY = -2.0
    STUCK_PENALTY = -3.0
    GOAL_REWARD = +2.5
    COIN_REWARD = 1.2
    SCORE_REWARD_SCALE = 0.015
    MAX_SCORE_REWARD = 2.5
    
    LR_INNER_ADAM = 0.0005
    LR_LIQUID_PLASTICITY = 2.5e-6
    LAMBDA_TRACE = 0.94
    ELIGIBILITY_DECAY = 0.98
    LAYER_NORM_EPSILON = 1e-8
    
    CURRICULUM_WARMUP_EPISODES = 100
    CURRICULUM_WINDOW = 20
    CURRICULUM_INCREMENT_SUCCESS = 10
    CURRICULUM_DECREMENT_FAIL = 5
    CURRICULUM_MIN_OFFSET = 15
    
    PRIORITY_ALPHA = 0.6
    PRIORITY_BETA_START = 0.4
    PRIORITY_BETA_FRAMES = 100000
    PRIORITY_EPSILON = 1e-6
    
    CORPUS_CALLOSUM_HEADS = 8
    CORPUS_CALLOSUM_DROPOUT = 0.1
    
    CHECKPOINT_DIR = "checkpoints_mario_v2"
    CHECKPOINT_EVERY = 10

    VISUAL_FEATURE_DIM = 128
    
    PATHWAY_ENTROPY_PENALTY = 0.003
    MIN_PATHWAY_WEIGHT = 0.30

    METRICS_LOG_EVERY = 5
    PATHWAY_CONTRIBUTION_WINDOW = 30
    GRADIENT_FLOW_THRESHOLD = 1e-5
    SALIENCY_PERCENTILE_THRESHOLD = 92
    SALIENCY_BLUR_KERNEL = 5
    SALIENCY_MIN_FOCUS_AREA = 0.03

    SALIENCY_MIN_FOCUS_AREA = 0.03

    LIQUID_PLASTICITY_UPDATE_FREQ = 10
    LIQUID_PLASTICITY_MAX_GRAD_NORM = 0.1
    LIQUID_PLASTICITY_SCALE = 0.15
    ELIGIBILITY_TRACE_CLAMP = 10.0
    DECODER_INIT_SCALE = 0.01
    
    SAFEGRAD_FEATURE_CLAMP = 5.0
    SAFEGRAD_DECODER_CLAMP = 1.0

    
    FATIGUE_DECAY_RATE = 0.97
    FATIGUE_ACCUMULATION_RATE = 0.001

    EXPLORATION_DECAY_RATE = 0.985
    TARGET_UPDATE_TAU = 0.001
    TARGET_UPDATE_FREQUENCY = 100
    TARGET_HARD_UPDATE_FREQUENCY = 500 
    GAMMA_DISCOUNT = 0.99
    
    PRIORITY_CLIP_TD_ERROR = 5.0
    
    # Audio Feature Generation
    AUDIO_EVENT_DIM = 8
    AUDIO_TEMPORAL_WINDOW = 4
    AUDIO_NOISE_SCALE = 0.05
    AUDIO_EMBEDDING_DIM = 64
    

    SALIENCY_SHARPENING_POWER = 8.0
    

    AUDIO_SIGNIFICANCE_THRESHOLD = 0.5
    AUDIO_BOOST_FACTOR = 1.8
    EPISODIC_MEMORY_THRESHOLD = 0.65
    EPISODIC_MEMORY_DECAY = 0.9995
    SURPRISE_LEARNING_RATE_MULTIPLIER = 3.0
    AUDIO_VISUAL_FUSION_DIM = 64
    AUDIO_ATTENTION_HEADS = 4
    TEMPORAL_HISTORY_WINDOW = 4
    SEMANTIC_TEMPORAL_DIM = 32
    CROSS_MODAL_TEMPERATURE = 0.07
    VISUAL_AUDIO_ALIGNMENT_WEIGHT = 0.3
    CONTRASTIVE_MARGIN = 0.5
    LEFT_HEMISPHERE_LSTM_LAYERS = 2
    LEFT_HEMISPHERE_LSTM_HIDDEN = 128
    LEFT_HEMISPHERE_ATTENTION_HEADS = 4
    LEFT_HEMISPHERE_MEMORY_WINDOW = 16
    LEFT_HEMISPHERE_DROPOUT = 0.1
    LEFT_HEMISPHERE_VALUE_HEAD_DIM = 64
    ALLOW_GRADIENT_FLOW_TO_RIGHT = True
    GRADIENT_SCALING_FACTOR = 0.1




os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)


def log_detailed_metrics(agent, ep, scores, losses, accuracies):
    if ep % Config.METRICS_LOG_EVERY != 0:
        return
    
    if len(scores) == 0:
        return
    
    vis_grad = agent.corpus_callosum.pathway_grad_norms[0].item()
    sem_grad = agent.corpus_callosum.pathway_grad_norms[2].item()
    
    if np.isnan(vis_grad) or np.isinf(vis_grad):
        vis_grad = 0.0
    if np.isnan(sem_grad) or np.isinf(sem_grad):
        sem_grad = 0.0
    
    total_grad = vis_grad + sem_grad + 1e-10
    vis_contribution = vis_grad / total_grad
    sem_contribution = sem_grad / total_grad
    
    visual_collapsed = vis_contribution < Config.GRADIENT_FLOW_THRESHOLD
    semantic_dominant = sem_contribution > 0.85
    
    recent_scores = scores[-10:] if len(scores) >= 10 else scores
    avg_reward = np.mean(recent_scores) if len(recent_scores) > 0 else 0.0
    
    goal_success = np.mean([s for _, s in agent.goal_history]) if len(agent.goal_history) > 0 else 0.0
    
    td_loss_safe = agent.last_td_loss if np.isfinite(agent.last_td_loss) else 0.0
    recon_loss_safe = agent.last_recon_loss if np.isfinite(agent.last_recon_loss) else 0.0
    entropy_pen_safe = agent.last_entropy_penalty if np.isfinite(agent.last_entropy_penalty) else 0.0
    total_loss_safe = agent.last_loss_inner if np.isfinite(agent.last_loss_inner) else 0.0
    
    memory_usage = agent.left_hemisphere.memory_filled.item()
    lstm_activity = agent.left_hemisphere.lstm_hidden.norm().item()
    
    print(f"\n{'='*60}")
    print(f"📊 MÉTRICAS DETALLADAS - Episodio {ep}")
    print(f"{'='*60}")
    print(f"🧠 Pathway Contributions:")
    print(f"   Visual:    {vis_contribution*100:6.2f}% {'⚠️ COLAPSADO' if visual_collapsed else '✅'}")
    print(f"   Semantic:  {sem_contribution*100:6.2f}% {'⚠️ DOMINANTE' if semantic_dominant else '✅'}")
    print(f"\n🔬 Loss Components:")
    print(f"   TD Loss:       {td_loss_safe:.6f}")
    print(f"   Recon Loss:    {recon_loss_safe:.6f}")
    print(f"   Entropy Pen:   {entropy_pen_safe:.6f}")
    print(f"   Total:         {total_loss_safe:.6f}")
    print(f"\n🎯 Performance:")
    print(f"   Avg Reward (10): {avg_reward:.2f}")
    print(f"   Goal Success:    {goal_success*100:.1f}%")
    print(f"\n🧪 Left Hemisphere (Executive):")
    print(f"   Memory Buffer:   {'FILLED ✅' if memory_usage else 'FILLING... ⏳'}")
    print(f"   LSTM Activity:   {lstm_activity:.4f}")
    print(f"{'='*60}\n")


class AudioFeatureGenerator(nn.Module):
    def __init__(self, device="cpu"):
        super(AudioFeatureGenerator, self).__init__()
        self.device = device
        self.event_history = deque(maxlen=Config.AUDIO_TEMPORAL_WINDOW)
        
        self.visual_context_encoder = nn.Sequential(
            nn.Linear(Config.VISUAL_FEATURE_DIM, Config.AUDIO_VISUAL_FUSION_DIM, dtype=torch.float32, device=device),
            nn.LayerNorm(Config.AUDIO_VISUAL_FUSION_DIM, dtype=torch.float32, device=device),
            nn.Tanh()
        )
        
        self.event_encoder = nn.Sequential(
            nn.Linear(Config.AUDIO_EVENT_DIM, Config.AUDIO_EMBEDDING_DIM, dtype=torch.float32, device=device),
            nn.Tanh(),
            nn.LayerNorm(Config.AUDIO_EMBEDDING_DIM, dtype=torch.float32, device=device),
            nn.Linear(Config.AUDIO_EMBEDDING_DIM, 128, dtype=torch.float32, device=device)
        )
        
        self.temporal_conv = nn.Conv1d(
            in_channels=Config.AUDIO_EVENT_DIM,
            out_channels=64,
            kernel_size=min(3, Config.AUDIO_TEMPORAL_WINDOW),
            padding=1,
            dtype=torch.float32,
            device=device
        )
        
        self.visual_audio_attention = nn.MultiheadAttention(
            embed_dim=Config.AUDIO_VISUAL_FUSION_DIM,
            num_heads=Config.AUDIO_ATTENTION_HEADS,
            batch_first=True,
            dtype=torch.float32,
            device=device
        )
        
        self.event_signature_encoder = nn.ModuleDict({
            'powerup': nn.Linear(1, 16, dtype=torch.float32, device=device),
            'coin': nn.Linear(1, 16, dtype=torch.float32, device=device),
            'enemy_defeat': nn.Linear(1, 16, dtype=torch.float32, device=device),
            'block_hit': nn.Linear(1, 16, dtype=torch.float32, device=device),
            'secret_found': nn.Linear(1, 16, dtype=torch.float32, device=device)
        })
        
        self.signature_fusion = nn.Linear(16 * 5, 64, dtype=torch.float32, device=device)
        
        self.fusion = nn.Sequential(
            nn.Linear(128 + 64 + 64 + Config.AUDIO_VISUAL_FUSION_DIM, 128, dtype=torch.float32, device=device),
            nn.LayerNorm(128, dtype=torch.float32, device=device)
        )
        
        nn.init.xavier_uniform_(self.visual_context_encoder[0].weight, gain=1.0)
        nn.init.xavier_uniform_(self.event_encoder[0].weight, gain=0.5)
        nn.init.xavier_uniform_(self.event_encoder[3].weight, gain=0.5)
        nn.init.xavier_uniform_(self.temporal_conv.weight, gain=0.5)
        nn.init.xavier_uniform_(self.signature_fusion.weight, gain=0.5)
        nn.init.xavier_uniform_(self.fusion[0].weight, gain=0.5)
        
        for encoder in self.event_signature_encoder.values():
            nn.init.xavier_uniform_(encoder.weight, gain=1.5)
            nn.init.zeros_(encoder.bias)
        
        self.register_buffer('prev_x_pos', torch.tensor(0.0))
        self.register_buffer('prev_y_pos', torch.tensor(0.0))
        self.register_buffer('prev_score', torch.tensor(0.0))
        self.register_buffer('prev_coins', torch.tensor(0.0))
        self.register_buffer('prev_life', torch.tensor(2.0))
        
        self.register_buffer('last_block_hit_x', torch.tensor(-1000.0))
        self.register_buffer('last_enemy_defeat_score', torch.tensor(0.0))
        
        self.to(device)
    
    def extract_event_features(self, info):
        x_pos = float(info.get('x_pos', 0))
        y_pos = float(info.get('y_pos', 0))
        score = float(info.get('score', 0))
        coins = float(info.get('coins', 0))
        life = float(info.get('life', 2))
        time_left = float(info.get('time', 400))
        
        coin_event = float(coins > self.prev_coins.item())
        score_diff = score - self.prev_score.item()
        
        powerup_event = 0.0
        if score_diff > Config.POWER_UP_THRESHOLD and coins == self.prev_coins.item():
            powerup_event = 1.0
        elif score_diff >= 1000 and score_diff % 1000 == 0:
            powerup_event = 1.0
        
        enemy_defeat_event = 0.0
        if 50 <= score_diff <= 500 and score_diff not in [200, 1000]:
            enemy_defeat_event = 1.0
            self.last_enemy_defeat_score.copy_(torch.tensor(score))
        
        block_hit_event = 0.0
        if abs(x_pos - self.last_block_hit_x.item()) > 5:
            if 180 <= score_diff <= 220:
                block_hit_event = 1.0
                self.last_block_hit_x.copy_(torch.tensor(x_pos))
        
        secret_found_event = 0.0
        if score_diff > 1000 and (score_diff % 1000 != 0):
            secret_found_event = 1.0
        
        damage_event = float(life < self.prev_life.item())
        
        x_velocity = np.clip((x_pos - self.prev_x_pos.item()) / 10.0, -1.0, 1.0)
        y_velocity = np.clip((y_pos - self.prev_y_pos.item()) / 10.0, -1.0, 1.0)
        
        features = torch.tensor([
            coin_event,
            powerup_event,
            enemy_defeat_event,
            block_hit_event,
            secret_found_event,
            damage_event,
            x_velocity,
            y_velocity
        ], dtype=torch.float32, device=self.device)
        
        self.prev_x_pos.copy_(torch.tensor(x_pos))
        self.prev_y_pos.copy_(torch.tensor(y_pos))
        self.prev_score.copy_(torch.tensor(score))
        self.prev_coins.copy_(torch.tensor(coins))
        self.prev_life.copy_(torch.tensor(life))
        
        return features
    
    def forward(self, info, visual_context=None):
        event_features = self.extract_event_features(info)
        
        self.event_history.append(event_features.detach().cpu().numpy())
        
        event_encoded = self.event_encoder(event_features.unsqueeze(0))
        
        if len(self.event_history) >= 2:
            history_array = np.array(list(self.event_history))
            history_tensor = torch.from_numpy(history_array).to(self.device).T.unsqueeze(0)
            temporal_features = self.temporal_conv(history_tensor)
            temporal_pooled = torch.mean(temporal_features, dim=2)
        else:
            temporal_pooled = torch.zeros(1, 64, dtype=torch.float32, device=self.device)
        
        signature_features = []
        event_types = ['powerup', 'coin', 'enemy_defeat', 'block_hit', 'secret_found']
        for i, event_type in enumerate(event_types):
            event_value = event_features[i if i < 5 else 0].unsqueeze(0).unsqueeze(0)
            signature = self.event_signature_encoder[event_type](event_value)
            signature_features.append(signature)
        
        signature_concat = torch.cat(signature_features, dim=-1)
        signature_fused = self.signature_fusion(signature_concat)
        
        if visual_context is not None:
            if not isinstance(visual_context, torch.Tensor):
                visual_context = torch.tensor(visual_context, dtype=torch.float32, device=self.device)
            
            if visual_context.dim() == 1:
                visual_context = visual_context.unsqueeze(0)
            
            visual_encoded = self.visual_context_encoder(visual_context)
            
            audio_query = event_encoded[:, :Config.AUDIO_VISUAL_FUSION_DIM]
            if audio_query.shape[-1] < Config.AUDIO_VISUAL_FUSION_DIM:
                audio_query = torch.nn.functional.pad(
                    audio_query, 
                    (0, Config.AUDIO_VISUAL_FUSION_DIM - audio_query.shape[-1])
                )
            
            audio_query = audio_query.unsqueeze(1)
            visual_key_value = visual_encoded.unsqueeze(1)
            
            attended_audio, attention_weights = self.visual_audio_attention(
                audio_query, visual_key_value, visual_key_value
            )
            attended_audio = attended_audio.squeeze(1)
        else:
            attended_audio = torch.zeros(1, Config.AUDIO_VISUAL_FUSION_DIM, dtype=torch.float32, device=self.device)
        
        combined = torch.cat([event_encoded, temporal_pooled, signature_fused, attended_audio], dim=-1)
        
        noise = torch.randn_like(combined) * Config.AUDIO_NOISE_SCALE
        audio_output = self.fusion(combined + noise)
        
        audio_output = torch.clamp(
            audio_output,
            -Config.SAFEGRAD_FEATURE_CLAMP,
            Config.SAFEGRAD_FEATURE_CLAMP
        )
        
        return audio_output.squeeze(0)
    
    def reset(self):
        self.event_history.clear()
        self.prev_x_pos.zero_()
        self.prev_y_pos.zero_()
        self.prev_score.zero_()
        self.prev_coins.zero_()
        self.prev_life.fill_(2.0)
        self.last_block_hit_x.fill_(-1000.0)
        self.last_enemy_defeat_score.zero_()



class FrameSkip(gym.Wrapper):
    def __init__(self, env, skip=Config.FRAMESKIP):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        combined_info = {
            'score': 0, 'coins': 0, 'time': 400, 'x_pos': 0, 'y_pos': 0, 
            'life': 2, 'stage': 1, 'world': 1
        }
        
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            
            
            for key in combined_info:
                if key in info:
                    if key in ['score', 'coins', 'x_pos', 'y_pos']:
                        combined_info[key] = max(combined_info[key], info[key])
                    elif key == 'time':
                        combined_info[key] = min(combined_info[key], info[key])
                    else:
                        combined_info[key] = info[key]
            
            if done:
                break
        
        return obs, total_reward, done, combined_info

class StuckMonitor(gym.Wrapper):
    def __init__(self, env, stuck_limit=Config.STUCK_LIMIT, inactivity_limit=Config.INACTIVITY_LIMIT):
        super().__init__(env)
        self.stuck_limit = stuck_limit
        self.inactivity_limit = inactivity_limit
        self.reset_stats()

    def reset_stats(self):
        self.max_x = 0
        self.stuck_steps = 0
        self.total_steps = 0
        self.last_score = 0
        self.last_time = 0
        self.last_coins = 0
        self.last_life = 2
        self.inactive_steps = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.total_steps += 1

        x = info.get('x_pos', 0)
        life = info.get('life', 2)

        
        if x > self.max_x:
            self.max_x = x
            self.stuck_steps = 0
        else:
            self.stuck_steps += 1
            info['stuck_steps'] = self.stuck_steps

        
        active = (info['score'] > self.last_score or
                  info['coins'] > self.last_coins or
                  info['time'] < self.last_time)
        self.inactive_steps = 0 if active else self.inactive_steps + 1
        info['inactive_steps'] = self.inactive_steps

        self.last_score = info['score']
        self.last_time = info['time']
        self.last_coins = info['coins']
        self.last_life = life

        
        if self.stuck_steps >= self.stuck_limit:
            done = True
        elif self.inactive_steps >= self.inactivity_limit:
            done = True
        elif self.total_steps >= Config.MAX_AGENT_STEPS:
            done = True
        elif life == 0:
            done = True

        return obs, reward, done, info

    def reset(self, **kwargs):
        
        result = self.env.reset(**kwargs)
        
        if isinstance(result, tuple):
            obs, info = result
        else:
            
            obs = result
            info = {}
        
        self.reset_stats()
        
        
        self.max_x = info.get('x_pos', 0)
        self.last_score = info.get('score', 0)
        self.last_time = info.get('time', 400)
        self.last_coins = info.get('coins', 0)
        self.last_life = info.get('life', 2)
        
        return obs, info


def preprocess_frame(frame):
    """Versión ultra-robusta: nunca devuelve NaN/inf."""
    try:
        if frame is None or frame.size == 0:
            return np.zeros((84, 84), dtype=np.float32)

        # Forzar uint8
        frame = np.clip(frame, 0, 255).astype(np.uint8)

        # Gris
        if len(frame.shape) == 2:
            gray = frame
        elif frame.shape[-1] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:  # canales extra
            gray = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2GRAY)

        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

        # ── FIX: eliminar NaN/inf producidos por resize o conversión ──
        resized = np.nan_to_num(resized, nan=0.0, posinf=0.0, neginf=0.0)

        return resized.astype(np.float32) / 255.0
    except Exception as exc:
        # Cualquier error → frame cero
        print(f"[preprocess_frame] EXCEPCIÓN: {exc}")
        return np.zeros((84, 84), dtype=np.float32)

def stack_frames(stacked, frame, is_new):
    frame = preprocess_frame(frame)
    
    if is_new:
        stacked = np.repeat(frame[None, :, :], 4, axis=0)
    else:
        stacked = np.concatenate([stacked[1:], frame[None, :, :]], axis=0)
    
    
    if stacked.shape != (4, 84, 84):
        fixed = np.zeros((4, 84, 84), dtype=np.float32)
        
        s0, s1, s2 = min(stacked.shape[0], 4), min(stacked.shape[1], 84), min(stacked.shape[2], 84)
        fixed[:s0, :s1, :s2] = stacked[:s0, :s1, :s2]
        stacked = fixed
    stacked = np.nan_to_num(stacked, nan=0.0, posinf=1.0, neginf=0.0)
    return stacked.astype(np.float32)



class VisualFeatureExtractor(nn.Module):
    def __init__(self, device="cpu"):
        super(VisualFeatureExtractor, self).__init__()
        self.device = device
        
        self.pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=4, stride=4),
            nn.AvgPool2d(kernel_size=3, stride=3),
        )
        
        self.conv = nn.Conv2d(4, 32, kernel_size=3, padding=1, dtype=torch.float32, device=device)
        
        self.attention_conv = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1, dtype=torch.float32, device=device),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1, dtype=torch.float32, device=device),
            nn.Sigmoid()
        )
        
        self.flattened_dim = 32 * 7 * 7
        
        self.feature_proj = nn.Linear(
            self.flattened_dim, 
            Config.VISUAL_FEATURE_DIM, 
            dtype=torch.float32, 
            device=device
        )
        
        self.decoder_proj = nn.Linear(
            self.flattened_dim,
            Config.VISUAL_FEATURE_DIM,
            dtype=torch.float32,
            device=device
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(Config.VISUAL_FEATURE_DIM, 512, dtype=torch.float32, device=device),
            nn.ReLU(),
            nn.LayerNorm(512, dtype=torch.float32, device=device),
            nn.Linear(512, 84 * 84, dtype=torch.float32, device=device),
        )

        nn.init.xavier_uniform_(self.decoder[0].weight, gain=Config.DECODER_INIT_SCALE)
        nn.init.zeros_(self.decoder[0].bias)
        nn.init.xavier_uniform_(self.decoder[3].weight, gain=Config.DECODER_INIT_SCALE)
        nn.init.zeros_(self.decoder[3].bias)
        
        nn.init.xavier_uniform_(self.decoder_proj.weight, gain=Config.DECODER_INIT_SCALE)
        nn.init.zeros_(self.decoder_proj.bias)
        
        nn.init.xavier_uniform_(self.attention_conv[0].weight, gain=1.0)
        nn.init.zeros_(self.attention_conv[0].bias)
        nn.init.xavier_uniform_(self.attention_conv[2].weight, gain=1.0)
        nn.init.zeros_(self.attention_conv[2].bias)
                
        self.to(device)
        
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        pooled = self.pool(x)
        conv_out = torch.relu(self.conv(pooled))
        
        attention_map = self.attention_conv(conv_out)
        
        attended_features = conv_out * attention_map
        
        flat = attended_features.view(batch_size, -1)
        flat = torch.clamp(flat, -Config.SAFEGRAD_FEATURE_CLAMP, Config.SAFEGRAD_FEATURE_CLAMP)
        
        features = self.feature_proj(flat)
        features = torch.clamp(features, -Config.SAFEGRAD_FEATURE_CLAMP, Config.SAFEGRAD_FEATURE_CLAMP)
        
        decoder_features = self.decoder_proj(flat.detach())
        decoder_features = torch.clamp(decoder_features, -Config.SAFEGRAD_DECODER_CLAMP, Config.SAFEGRAD_DECODER_CLAMP)
        
        decoder_pre = self.decoder(decoder_features).view(batch_size, 84, 84)
        decoder_pre = torch.clamp(decoder_pre, -10.0, 10.0)
        reconstructed = torch.sigmoid(decoder_pre)
        
        return features, reconstructed

# ==============================================================================
# STABLELIQUIDNEURON
# ==============================================================================
class StableLiquidNeuron(nn.Module):
    def __init__(self, in_dim, out_dim, device="cpu"):
        super(StableLiquidNeuron, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        
        self.W_slow = nn.Parameter(torch.randn(in_dim, out_dim, dtype=torch.float32, device=device) * np.sqrt(2.0 / in_dim))
        
        self.register_buffer('W_fast_short', 0.0001 * torch.randn(out_dim, in_dim, dtype=torch.float32, device=device))
        self.register_buffer('W_fast_long', 0.00005 * torch.randn(out_dim, in_dim, dtype=torch.float32, device=device))
        
        self.register_buffer('homeostasis', torch.tensor(1.0, dtype=torch.float32, device=device))
        self.register_buffer('metabolism', torch.tensor(0.6, dtype=torch.float32, device=device))
        self.register_buffer('fatigue', torch.tensor(0.0, dtype=torch.float32, device=device))
        
        self.register_buffer('eligibility_trace', torch.zeros(in_dim, out_dim, dtype=torch.float32, device=device))
        self.register_buffer('trace_momentum', torch.zeros(in_dim, out_dim, dtype=torch.float32, device=device))
        self.register_buffer('td_history', torch.zeros(Config.PATHWAY_CONTRIBUTION_WINDOW, dtype=torch.float32, device=device))
        self.register_buffer('td_index', torch.tensor(0, dtype=torch.long, device=device))
        
        self.proj_w = nn.Parameter(torch.randn(out_dim * 3, out_dim, dtype=torch.float32, device=device) * 0.01)
        
        self.layer_norm_slow = nn.LayerNorm(out_dim, eps=Config.LAYER_NORM_EPSILON, device=device)
        self.layer_norm_fast_short = nn.LayerNorm(out_dim, eps=Config.LAYER_NORM_EPSILON, device=device)
        self.layer_norm_fast_long = nn.LayerNorm(out_dim, eps=Config.LAYER_NORM_EPSILON, device=device)
        
        self.sensitivity = 0.5
        
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        
        x = x.to(self.device)
        
        if x.shape[-1] != self.in_dim:
            if x.shape[-1] < self.in_dim:
                x = torch.nn.functional.pad(x, (0, self.in_dim - x.shape[-1]))
            else:
                x = x[..., :self.in_dim]
        
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        slow_out = x @ self.W_slow
        fast_short = x @ self.W_fast_short.T
        fast_long = x @ self.W_fast_long.T
        
        slow_out = self.layer_norm_slow(slow_out)
        fast_short = self.layer_norm_fast_short(fast_short)
        fast_long = self.layer_norm_fast_long(fast_long)
        
        gate_short = 0.05 + 0.15 * self.sensitivity * self.homeostasis
        gate_long = 0.02 + 0.08 * self.metabolism
        
        combined = torch.cat([slow_out, gate_short * fast_short, gate_long * fast_long], dim=-1)
        
        output = combined @ self.proj_w
        
        if torch.isnan(output).any() or torch.isinf(output).any():
            output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
        
        with torch.no_grad():
            output_norm = torch.norm(output, dim=-1).mean()
            if torch.isfinite(output_norm):
                self.homeostasis.copy_(torch.clamp(torch.tanh(2.0 - torch.abs(output_norm - 1.0)), 0.5, 1.0))
        
        return output
    
    def compute_plasticity_gradient(self, x, output, td_error):
        if not isinstance(td_error, torch.Tensor):
            td_error = torch.tensor(td_error, dtype=torch.float32, device=self.device)
        
        idx = self.td_index % Config.PATHWAY_CONTRIBUTION_WINDOW
        self.td_history[idx] = torch.abs(td_error)
        self.td_index += 1
        
        if self.td_index >= Config.PATHWAY_CONTRIBUTION_WINDOW:
            td_variance = torch.var(self.td_history)
            td_mean = torch.mean(self.td_history)
            
            if td_mean > 2.0:
                trace_boost = 0.5
            else:
                trace_boost = 1.0 + torch.clamp(td_variance * 10.0, 0.0, 2.0)
        else:
            trace_boost = 1.0
        
        norm_td = torch.tanh(td_error * 0.1).to(self.device)
        
        x_detached = x.detach()
        output_detached = output.detach()
        
        x_flat = x_detached.view(-1, self.in_dim)
        out_flat = output_detached.view(-1, self.out_dim)
        hebbian_update = torch.bmm(x_flat.unsqueeze(2), out_flat.unsqueeze(1)).mean(0)
        
        hebbian_norm = torch.norm(hebbian_update)
        if hebbian_norm > 1.0:
            hebbian_update = hebbian_update / hebbian_norm
        
        self.trace_momentum.mul_(0.9).add_(hebbian_update * 0.1)
        
        amplified_update = hebbian_update * trace_boost + self.trace_momentum * 0.3
        
        self.eligibility_trace.mul_(Config.LAMBDA_TRACE).add_(amplified_update)
        
        self.eligibility_trace.clamp_(-Config.ELIGIBILITY_TRACE_CLAMP, Config.ELIGIBILITY_TRACE_CLAMP)
        
        plastic_grad = -(self.eligibility_trace * norm_td).clamp(
            -Config.LIQUID_PLASTICITY_MAX_GRAD_NORM, 
            Config.LIQUID_PLASTICITY_MAX_GRAD_NORM
        )
        
        plastic_grad = plastic_grad * Config.LIQUID_PLASTICITY_SCALE
        
        return plastic_grad


        
    def post_step_update(self):
        self.eligibility_trace.mul_(Config.ELIGIBILITY_DECAY)
        
        trace_norm = torch.norm(self.eligibility_trace)
        if trace_norm < Config.GRADIENT_FLOW_THRESHOLD:
            self.eligibility_trace.mul_(1.1)



class RightHemisphere(nn.Module):
    def __init__(self, input_dim=Config.VISUAL_FEATURE_DIM, output_dim=128, aux_dim=13, device="cpu"):
        super(RightHemisphere, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.aux_dim = aux_dim
        self.device = device
        
        self.visual_extractor = VisualFeatureExtractor(device=device)
        
        self.visual_liquid = StableLiquidNeuron(input_dim, output_dim, device=device)
        self.semantic_liquid = StableLiquidNeuron(aux_dim + 1, output_dim, device=device)
        
        self.goal_encoder = nn.Sequential(
            nn.Linear(1, 16, dtype=torch.float32, device=device),
            nn.Tanh(),
            nn.Linear(16, 1, dtype=torch.float32, device=device)
        )
        
        self.audio_generator = AudioFeatureGenerator(device=device)
        
        self.audio_liquid = StableLiquidNeuron(128, output_dim, device=device)
        
        self.to(device)

                    
    def forward(self, stacked_frame, aux_features, goal_x=None, current_x=None, info=None, visualization_mode=False):
        print(f"[DEBUG RH] Start forward")
        
        if not isinstance(stacked_frame, torch.Tensor):
            stacked_frame = torch.tensor(stacked_frame, dtype=torch.float32, device=self.device)
        
        stacked_frame = stacked_frame.to(self.device)
        
        if stacked_frame.shape == (4, 84, 84):
            stacked_frame = stacked_frame.unsqueeze(0)
        elif len(stacked_frame.shape) == 3:
            stacked_frame = stacked_frame.unsqueeze(0)
        
        batch_size = stacked_frame.shape[0]
        print(f"[DEBUG RH] Batch size: {batch_size}")
        
        print(f"[DEBUG RH] Visual extractor forward...")
        visual_features, visual_reconstruction = self.visual_extractor(stacked_frame)
        
        print(f"[DEBUG RH] Visual liquid forward...")
        if visualization_mode:
            visual_features_for_liquid = torch.clamp(visual_features, -Config.SAFEGRAD_FEATURE_CLAMP, Config.SAFEGRAD_FEATURE_CLAMP)
        else:
            visual_features_for_liquid = visual_features.detach()
            visual_features_for_liquid = torch.clamp(visual_features_for_liquid, -Config.SAFEGRAD_FEATURE_CLAMP, Config.SAFEGRAD_FEATURE_CLAMP)
        
        visual_out = self.visual_liquid(visual_features_for_liquid)
        print(f"[DEBUG RH] Visual liquid done")
        
        print(f"[DEBUG RH] Goal encoding...")
        if goal_x is not None and current_x is not None:
            goal_delta = torch.tensor([[float(goal_x - current_x) / 100.0]], dtype=torch.float32, device=self.device)
            goal_delta = torch.clamp(goal_delta, -10.0, 10.0)
            goal_emb = self.goal_encoder(goal_delta)
            goal_emb = goal_emb.expand(batch_size, -1)
        else:
            goal_emb = torch.zeros(batch_size, 1, dtype=torch.float32, device=self.device)
        print(f"[DEBUG RH] Goal encoding done")
        
        print(f"[DEBUG RH] Aux tensor processing...")
        aux_tensor = aux_features if isinstance(aux_features, torch.Tensor) else torch.tensor(aux_features, dtype=torch.float32, device=self.device)
        if aux_tensor.ndim == 1:
            aux_tensor = aux_tensor.unsqueeze(0)
        aux_tensor = aux_tensor.to(self.device)
        
        if aux_tensor.shape[0] != batch_size:
            aux_tensor = aux_tensor.expand(batch_size, -1)
        
        aux_tensor = torch.clamp(aux_tensor, -1.0, 2.0)
        
        aux_with_goal = torch.cat([aux_tensor[:, :self.aux_dim], goal_emb], dim=1)
        print(f"[DEBUG RH] Semantic liquid forward...")
        semantic_out = self.semantic_liquid(aux_with_goal)
        print(f"[DEBUG RH] Semantic liquid done")
        
        print(f"[DEBUG RH] Audio features...")
        if info is not None:
            audio_features = self.audio_generator(info, visual_context=visual_features.detach())
            audio_out = self.audio_liquid(audio_features.unsqueeze(0) if audio_features.dim() == 1 else audio_features)
            if audio_out.shape[0] != batch_size:
                audio_out = audio_out.expand(batch_size, -1)
        else:
            audio_out = torch.zeros(batch_size, self.output_dim, dtype=torch.float32, device=self.device)
        print(f"[DEBUG RH] Audio done")
        
        print(f"[DEBUG RH] Returning...")
        return visual_out, audio_out, semantic_out, visual_features, aux_tensor, visual_reconstruction
            
            

        

    def forward_legacy(self, stacked_frame, aux_features, goal_x=None, current_x=None):
        """Wrapper para compatibilidad con código que espera 5 retornos"""
        visual_out, audio_sim, semantic_out, visual_features, aux_tensor, visual_reconstruction = self.forward(
            stacked_frame, aux_features, goal_x, current_x
        )
        return visual_out, audio_sim, semantic_out, visual_features, aux_tensor


# ==============================================================================
# CORPUSCALLOSUM
# ==============================================================================
class CorpusCallosum(nn.Module):
    def __init__(self, dim=128):
        super(CorpusCallosum, self).__init__()
        self.dim = dim
        
        self.visual_proj = nn.Linear(dim, dim, dtype=torch.float32)
        self.audio_proj = nn.Linear(dim, dim, dtype=torch.float32)
        self.semantic_proj = nn.Linear(dim, dim, dtype=torch.float32)
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=Config.CORPUS_CALLOSUM_HEADS,
            dropout=Config.CORPUS_CALLOSUM_DROPOUT,
            batch_first=True,
            dtype=torch.float32
        )
        
        self.context_gate = nn.Sequential(
            nn.Linear(dim * 3, 64, dtype=torch.float32),
            nn.LayerNorm(64),
            nn.Tanh(),
            nn.Linear(64, 3, dtype=torch.float32),
            nn.Softmax(dim=-1)
        )
        
        self.register_buffer('visual_fatigue', torch.tensor(0.0))
        self.register_buffer('audio_fatigue', torch.tensor(0.0))
        self.register_buffer('semantic_fatigue', torch.tensor(0.0))
        
        self.register_buffer('pathway_grad_norms', torch.zeros(3))
        self.register_buffer('pathway_usage_history', torch.zeros(3, Config.PATHWAY_CONTRIBUTION_WINDOW))
        self.register_buffer('history_index', torch.tensor(0, dtype=torch.long))
        
        self.fatigue_decay = 0.95
        self.max_fatigue = 0.8
        

    def forward(self, visual_features, audio_features, semantic_features, td_error=None):
        device = visual_features.device
        
        if not visual_features.requires_grad:
            visual_features = visual_features.detach().requires_grad_(True)
        if not audio_features.requires_grad:
            audio_features = audio_features.detach().requires_grad_(True)
        if not semantic_features.requires_grad:
            semantic_features = semantic_features.detach().requires_grad_(True)
        
        if td_error is not None:
            if not isinstance(td_error, torch.Tensor):
                td_error = torch.tensor(td_error, dtype=torch.float32, device=device)
            fatigue_mod = 1.0 - torch.sigmoid(td_error * 5)
        else:
            fatigue_mod = 1.0
        
        vis_fatigue_value = torch.clamp(self.visual_fatigue * fatigue_mod, 0.0, self.max_fatigue).item()
        aud_fatigue_value = torch.clamp(self.audio_fatigue * fatigue_mod, 0.0, self.max_fatigue).item()
        sem_fatigue_value = torch.clamp(self.semantic_fatigue * fatigue_mod, 0.0, self.max_fatigue).item()
        
        visual_proj = self.visual_proj(visual_features)
        audio_proj = self.audio_proj(audio_features)
        semantic_proj = self.semantic_proj(semantic_features)
        
        visual_norm = torch.nn.functional.normalize(visual_proj, p=2, dim=-1)
        audio_norm = torch.nn.functional.normalize(audio_proj, p=2, dim=-1)
        semantic_norm = torch.nn.functional.normalize(semantic_proj, p=2, dim=-1)
        
        visual_audio_sim = torch.sum(visual_norm * audio_norm, dim=-1, keepdim=True)
        visual_semantic_sim = torch.sum(visual_norm * semantic_norm, dim=-1, keepdim=True)
        audio_semantic_sim = torch.sum(audio_norm * semantic_norm, dim=-1, keepdim=True)
        
        alignment_loss = (
            torch.clamp(Config.CONTRASTIVE_MARGIN - visual_audio_sim, min=0.0).mean() +
            torch.clamp(Config.CONTRASTIVE_MARGIN - visual_semantic_sim, min=0.0).mean() +
            torch.clamp(Config.CONTRASTIVE_MARGIN - audio_semantic_sim, min=0.0).mean()
        )
        
        modalities = torch.stack([visual_proj, audio_proj, semantic_proj], dim=1)
        
        attn_output, attn_weights = self.multihead_attn(
            modalities, modalities, modalities,
            need_weights=True, average_attn_weights=True
        )
        
        fatigue_weights = torch.tensor([
            1.0 - vis_fatigue_value * 0.5,
            1.0 - aud_fatigue_value * 0.5,
            1.0 - sem_fatigue_value * 0.5
        ], device=device, dtype=torch.float32)
        
        visual_gated = attn_output[:, 0, :] * fatigue_weights[0]
        audio_gated = attn_output[:, 1, :] * fatigue_weights[1]
        semantic_gated = attn_output[:, 2, :] * fatigue_weights[2]
        
        audio_norm_magnitude = torch.norm(audio_features, dim=-1, keepdim=True)
        audio_is_significant = (audio_norm_magnitude > Config.AUDIO_SIGNIFICANCE_THRESHOLD).float()
        
        audio_boost = 1.0 + audio_is_significant * Config.AUDIO_BOOST_FACTOR
        audio_gated = audio_gated * audio_boost
        
        combined_features = torch.cat([visual_gated, audio_gated, semantic_gated], dim=-1)
        
        raw_logits = self.context_gate[:-1](combined_features)
        
        with torch.no_grad():
            current_weights_detached = torch.sigmoid(raw_logits)
            
            idx = self.history_index % Config.PATHWAY_CONTRIBUTION_WINDOW
            self.pathway_usage_history[:, idx] = current_weights_detached.mean(dim=0)
            self.history_index += 1
            
            if self.history_index >= Config.PATHWAY_CONTRIBUTION_WINDOW:
                avg_usage = self.pathway_usage_history.mean(dim=1)
                pathway_variance = torch.var(avg_usage)
                
                if pathway_variance < Config.GRADIENT_FLOW_THRESHOLD:
                    boost_visual = 1.0 + max(0.0, Config.MIN_PATHWAY_WEIGHT - avg_usage[0].item()) * 1.5
                    boost_audio = 1.0 + max(0.0, Config.MIN_PATHWAY_WEIGHT - avg_usage[1].item()) * 2.5
                    boost_semantic = 1.0 + max(0.0, Config.MIN_PATHWAY_WEIGHT - avg_usage[2].item()) * 2.0
                    
                    adaptive_bias = torch.tensor([
                        boost_visual - 1.0, 
                        boost_audio - 1.0, 
                        boost_semantic - 1.0
                    ], dtype=torch.float32, device=device)
                else:
                    adaptive_bias = torch.zeros(3, dtype=torch.float32, device=device)
            else:
                adaptive_bias = torch.zeros(3, dtype=torch.float32, device=device)
        
        adjusted_logits = raw_logits + adaptive_bias.unsqueeze(0)
        
        context_weights = torch.sigmoid(adjusted_logits)
        
        context_weights_clamped = torch.clamp(context_weights, min=Config.MIN_PATHWAY_WEIGHT)
        context_weights_normalized = context_weights_clamped / (context_weights_clamped.sum(dim=-1, keepdim=True) + 1e-10)
        
        fused = (context_weights_normalized[:, 0:1] * visual_gated + 
                context_weights_normalized[:, 1:2] * audio_gated + 
                context_weights_normalized[:, 2:3] * semantic_gated)
        
        context_weights_safe = torch.clamp(context_weights_normalized, min=1e-10, max=1.0)
        log_weights = torch.log(context_weights_safe + 1e-10)
        log_weights = torch.nan_to_num(log_weights, nan=-10.0, posinf=0.0, neginf=-10.0)
        
        entropy = -(context_weights_safe * log_weights).sum(dim=-1).mean()
        entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)
        
        if not entropy.requires_grad:
            entropy = entropy * torch.ones_like(entropy, device=device, requires_grad=True)
        
        entropy_penalty = -Config.PATHWAY_ENTROPY_PENALTY * entropy
        
        total_penalty = entropy_penalty + Config.VISUAL_AUDIO_ALIGNMENT_WEIGHT * alignment_loss
        
        if not total_penalty.requires_grad:
            total_penalty = total_penalty * torch.ones_like(total_penalty, device=device, requires_grad=True)
        
        self.visual_fatigue.copy_(torch.clamp(
            torch.tensor(vis_fatigue_value * Config.FATIGUE_DECAY_RATE + Config.FATIGUE_ACCUMULATION_RATE * fatigue_mod), 
            0.0, self.max_fatigue
        ))
        self.audio_fatigue.copy_(torch.clamp(
            torch.tensor(aud_fatigue_value * Config.FATIGUE_DECAY_RATE + Config.FATIGUE_ACCUMULATION_RATE * 0.8 * fatigue_mod), 
            0.0, self.max_fatigue
        ))
        self.semantic_fatigue.copy_(torch.clamp(
            torch.tensor(sem_fatigue_value * Config.FATIGUE_DECAY_RATE + Config.FATIGUE_ACCUMULATION_RATE * 1.2 * fatigue_mod), 
            0.0, self.max_fatigue
        ))
        
        with torch.no_grad():
            if visual_gated.grad is not None:
                self.pathway_grad_norms[0] = visual_gated.grad.norm()
            else:
                self.pathway_grad_norms[0] = torch.norm(visual_gated)
            
            if audio_gated.grad is not None:
                self.pathway_grad_norms[1] = audio_gated.grad.grad.norm()
            else:
                self.pathway_grad_norms[1] = torch.norm(audio_gated)
            
            if semantic_gated.grad is not None:
                self.pathway_grad_norms[2] = semantic_gated.grad.norm()
            else:
                self.pathway_grad_norms[2] = torch.norm(semantic_gated)

        info = {
            'visual': visual_gated.detach(),
            'audio': audio_gated.detach(),
            'semantic': semantic_gated.detach(),
            'fatigue': {
                'visual': vis_fatigue_value,
                'audio': aud_fatigue_value,
                'semantic': sem_fatigue_value
            },
            'attention': attn_weights.detach(),
            'context_weights': context_weights_normalized.detach(),
            'entropy_penalty': total_penalty.detach(),
            'alignment_loss': alignment_loss.detach(),
            'cross_modal_similarity': {
                'visual_audio': visual_audio_sim.mean().item(),
                'visual_semantic': visual_semantic_sim.mean().item(),
                'audio_semantic': audio_semantic_sim.mean().item()
            }
        }

        return fused, info, total_penalty

        



    def reset_fatigue(self):
        self.visual_fatigue.zero_()
        self.audio_fatigue.zero_()
        self.semantic_fatigue.zero_()
        self.history_index.zero_()

class PrioritizedReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        
    def __len__(self):
        return len(self.buffer)
    
    def add(self, experience, td_error=None):
        priority = (abs(td_error) + Config.PRIORITY_EPSILON) ** Config.PRIORITY_ALPHA if td_error is not None else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size, beta):
        if len(self.buffer) == 0:
            return [], [], []
        
        print(f"[DEBUG SAMPLE] Buffer len: {len(self.buffer)}, batch_size: {batch_size}")
        
        priorities = self.priorities[:len(self.buffer)]
        priorities = np.maximum(priorities, Config.PRIORITY_EPSILON)
        
        print(f"[DEBUG SAMPLE] Priorities shape: {priorities.shape}, sum: {priorities.sum()}")
        
        # FIX: Asegurar que las probabilidades sean válidas
        probs = priorities / (priorities.sum() + 1e-10)  # Agregar epsilon al denominador
        
        if np.isnan(probs).any() or np.isinf(probs).any() or probs.sum() == 0:
            print("[WARNING] Probs inválidas, usando distribución uniforme")
            probs = np.ones(len(self.buffer)) / len(self.buffer)
        
        print(f"[DEBUG SAMPLE] Probs sum: {probs.sum()}, calling np.random.choice...")
        
        try:
            indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), p=probs, replace=False)
            print(f"[DEBUG SAMPLE] Indices sampled: {len(indices)}")
        except Exception as e:
            print(f"[ERROR SAMPLE] np.random.choice failed: {e}")
            # Fallback: sampleo uniforme
            indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        
        samples = [self.buffer[idx] for idx in indices]
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= (weights.max() + 1e-10)  # Agregar epsilon
        
        print(f"[DEBUG SAMPLE] Returning {len(samples)} samples")
        return samples, indices, weights
    
    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = (abs(td_error) + Config.PRIORITY_EPSILON) ** Config.PRIORITY_ALPHA



class LeftHemisphere(nn.Module):
    def __init__(self, n_actions, input_dim=128, hidden_dim=128):
        super(LeftHemisphere, self).__init__()
        self.n_actions = n_actions
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        

        self.fc1 = nn.Linear(input_dim, hidden_dim, dtype=torch.float32)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32)
        self.fc3 = nn.Linear(hidden_dim, n_actions, dtype=torch.float32)
        
        
        self.register_buffer(
            'memory_buffer', 
            torch.zeros(100, input_dim, dtype=torch.float32)
        )
        self.register_buffer(
            'memory_index', 
            torch.tensor(0, dtype=torch.long)
        )
        self.register_buffer(
            'memory_filled', 
            torch.tensor(0, dtype=torch.long)
        )
        self.lstm_hidden_size = 64
        self.register_buffer(
            'lstm_hidden', 
            torch.zeros(1, self.lstm_hidden_size, dtype=torch.float32)
        )
        self.register_buffer(
            'lstm_cell', 
            torch.zeros(1, self.lstm_hidden_size, dtype=torch.float32)
        )

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
    
    def forward(self, x):
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        x = self.fc1(x)
        x = x / (1 + torch.exp(-1.75 * x))
        
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        x = self.fc2(x)
        x = x / (1 + torch.exp(-1.75 * x))
        
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        q = self.fc3(x)
        
        if torch.isnan(q).any() or torch.isinf(q).any():
            q = torch.nan_to_num(q, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return q.squeeze(0) if q.dim() > 1 else q
    
    def forward_with_cache(self, x):
        cache = {}
        z1 = self.fc1(x)
        a1 = z1 / (1 + torch.exp(-1.75 * z1))
        cache['z1'] = z1
        cache['a1'] = a1
        
        z2 = self.fc2(a1)
        a2 = z2 / (1 + torch.exp(-1.75 * z2))
        cache['z2'] = z2
        cache['a2'] = a2
        
        q = self.fc3(a2)
        cache['x'] = x
        
        return q.squeeze(0) if q.dim() > 1 else q, cache
    
    # ═══════════════════════════════════════════════════════════════
    # MÉTODO reset_memory()
    # ═══════════════════════════════════════════════════════════════
    def reset_memory(self):
        """Reinicia el buffer de memoria episódica del hemisferio izquierdo"""
        self.memory_buffer.zero_()
        self.memory_index.zero_()
        self.memory_filled.zero_()
        self.lstm_hidden.zero_()  
        self.lstm_cell.zero_()  


    def store_experience(self, x):
        """Almacena una experiencia en el buffer circular"""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.memory_buffer.device)
        
        if x.dim() == 1 and x.shape[0] == self.input_dim:
            idx = self.memory_index % self.memory_buffer.shape[0]
            self.memory_buffer[idx] = x.detach()
            self.memory_index += 1
    
            if self.memory_filled < self.memory_buffer.shape[0]:
                self.memory_filled += 1
        elif x.dim() == 2 and x.shape[-1] == self.input_dim:
            idx = self.memory_index % self.memory_buffer.shape[0]
            self.memory_buffer[idx] = x[0].detach()
            self.memory_index += 1
   
            if self.memory_filled < self.memory_buffer.shape[0]:
                self.memory_filled += 1




class EpisodicMemory(nn.Module):
    def __init__(self, device="cpu"):
        super(EpisodicMemory, self).__init__()
        self.device = device
        
        self.register_buffer('powerup_prototype', torch.zeros(128, dtype=torch.float32, device=device))
        self.register_buffer('powerup_count', torch.tensor(0, dtype=torch.long, device=device))
        
        self.register_buffer('enemy_defeat_prototype', torch.zeros(128, dtype=torch.float32, device=device))
        self.register_buffer('enemy_defeat_count', torch.tensor(0, dtype=torch.long, device=device))
        
        self.register_buffer('secret_prototype', torch.zeros(128, dtype=torch.float32, device=device))
        self.register_buffer('secret_count', torch.tensor(0, dtype=torch.long, device=device))
        
        self.register_buffer('pipe_prototype', torch.zeros(128, dtype=torch.float32, device=device))
        self.register_buffer('pipe_count', torch.tensor(0, dtype=torch.long, device=device))
        
        self.register_buffer('gap_jump_prototype', torch.zeros(128, dtype=torch.float32, device=device))
        self.register_buffer('gap_jump_count', torch.tensor(0, dtype=torch.long, device=device))
        
        self.memory_encoder = nn.Sequential(
            nn.Linear(128, 256, dtype=torch.float32, device=device),
            nn.LayerNorm(256, dtype=torch.float32, device=device),
            nn.Tanh(),
            nn.Linear(256, 128, dtype=torch.float32, device=device)
        )
        
        nn.init.xavier_uniform_(self.memory_encoder[0].weight, gain=1.0)
        nn.init.xavier_uniform_(self.memory_encoder[3].weight, gain=1.0)
        
        self.to(device)
    
    def store_episode(self, event_type, liquid_state):
        if not isinstance(liquid_state, torch.Tensor):
            liquid_state = torch.tensor(liquid_state, dtype=torch.float32, device=self.device)
        
        liquid_state = liquid_state.detach().flatten()
        
        if liquid_state.shape[0] != 128:
            if liquid_state.shape[0] < 128:
                liquid_state = torch.nn.functional.pad(liquid_state, (0, 128 - liquid_state.shape[0]))
            else:
                liquid_state = liquid_state[:128]
        
        encoded_state = self.memory_encoder(liquid_state.unsqueeze(0)).squeeze(0)
        
        if event_type == 'powerup':
            count = self.powerup_count.item()
            self.powerup_prototype.copy_(
                (self.powerup_prototype * count + encoded_state) / (count + 1)
            )
            self.powerup_count += 1
        elif event_type == 'enemy_defeat':
            count = self.enemy_defeat_count.item()
            self.enemy_defeat_prototype.copy_(
                (self.enemy_defeat_prototype * count + encoded_state) / (count + 1)
            )
            self.enemy_defeat_count += 1
        elif event_type == 'secret':
            count = self.secret_count.item()
            self.secret_prototype.copy_(
                (self.secret_prototype * count + encoded_state) / (count + 1)
            )
            self.secret_count += 1
        elif event_type == 'pipe':
            count = self.pipe_count.item()
            self.pipe_prototype.copy_(
                (self.pipe_prototype * count + encoded_state) / (count + 1)
            )
            self.pipe_count += 1
        elif event_type == 'gap_jump':
            count = self.gap_jump_count.item()
            self.gap_jump_prototype.copy_(
                (self.gap_jump_prototype * count + encoded_state) / (count + 1)
            )
            self.gap_jump_count += 1
    
    def retrieve_similar_episode(self, current_state):
        if not isinstance(current_state, torch.Tensor):
            current_state = torch.tensor(current_state, dtype=torch.float32, device=self.device)
        
        current_state = current_state.detach().flatten()
        
        if current_state.shape[0] != 128:
            if current_state.shape[0] < 128:
                current_state = torch.nn.functional.pad(current_state, (0, 128 - current_state.shape[0]))
            else:
                current_state = current_state[:128]
        
        encoded_current = self.memory_encoder(current_state.unsqueeze(0)).squeeze(0)
        
        similarities = {}
        if self.powerup_count > 0:
            similarities['powerup'] = torch.cosine_similarity(
                encoded_current.unsqueeze(0), 
                self.powerup_prototype.unsqueeze(0)
            ).item()
        if self.enemy_defeat_count > 0:
            similarities['enemy_defeat'] = torch.cosine_similarity(
                encoded_current.unsqueeze(0), 
                self.enemy_defeat_prototype.unsqueeze(0)
            ).item()
        if self.secret_count > 0:
            similarities['secret'] = torch.cosine_similarity(
                encoded_current.unsqueeze(0), 
                self.secret_prototype.unsqueeze(0)
            ).item()
        if self.pipe_count > 0:
            similarities['pipe'] = torch.cosine_similarity(
                encoded_current.unsqueeze(0), 
                self.pipe_prototype.unsqueeze(0)
            ).item()
        if self.gap_jump_count > 0:
            similarities['gap_jump'] = torch.cosine_similarity(
                encoded_current.unsqueeze(0), 
                self.gap_jump_prototype.unsqueeze(0)
            ).item()
        
        if len(similarities) > 0:
            best_match = max(similarities.items(), key=lambda x: x[1])
            if best_match[1] > Config.EPISODIC_MEMORY_THRESHOLD:
                return best_match[0], best_match[1]
        
        return None, 0.0



class TricameralMarioAgent(nn.Module):
    def __init__(self, n_actions, device="cpu"):
        super(TricameralMarioAgent, self).__init__()
        self.n_actions = n_actions
        self.memory = PrioritizedReplayBuffer(capacity=10000)
        self.goal_history = deque(maxlen=Config.GOAL_BUFFER_SIZE)
        self.last_q_values = np.zeros(n_actions, dtype=np.float32)
        self.device = device
        
        self.right_hemisphere = RightHemisphere(output_dim=128, aux_dim=5, device=device)
        self.corpus_callosum = CorpusCallosum(dim=128)
        self.left_hemisphere = LeftHemisphere(n_actions, input_dim=128)
        self.episodic_memory = EpisodicMemory(device=device)

        self.register_buffer('prev_score_for_events', torch.tensor(0.0))
        self.register_buffer('prev_coins_for_events', torch.tensor(0.0))
        self.register_buffer('prev_y_pos_for_events', torch.tensor(0.0))
        self.register_buffer('prev_x_pos_for_events', torch.tensor(0.0))
        self.target_right_hemisphere = deepcopy(self.right_hemisphere)
        self.target_corpus_callosum = deepcopy(self.corpus_callosum)
        self.target_left_hemisphere = deepcopy(self.left_hemisphere)
        
        for param in self.target_right_hemisphere.parameters():
            param.requires_grad = False
        for param in self.target_corpus_callosum.parameters():
            param.requires_grad = False
        for param in self.target_left_hemisphere.parameters():
            param.requires_grad = False
        
        self.optimizer = optim.Adam([
            {'params': self.right_hemisphere.visual_extractor.conv.parameters(), 'lr': Config.LR_INNER_ADAM},
            {'params': self.right_hemisphere.visual_extractor.feature_proj.parameters(), 'lr': Config.LR_INNER_ADAM},
            {'params': self.right_hemisphere.visual_liquid.parameters(), 'lr': Config.LR_INNER_ADAM},
            {'params': self.right_hemisphere.semantic_liquid.parameters(), 'lr': Config.LR_INNER_ADAM},
            {'params': self.right_hemisphere.audio_liquid.parameters(), 'lr': Config.LR_INNER_ADAM},
            {'params': self.right_hemisphere.audio_generator.parameters(), 'lr': Config.LR_INNER_ADAM * 0.5},
            {'params': self.right_hemisphere.goal_encoder.parameters(), 'lr': Config.LR_INNER_ADAM},
            {'params': self.corpus_callosum.parameters(), 'lr': Config.LR_INNER_ADAM},
            {'params': self.left_hemisphere.parameters(), 'lr': Config.LR_INNER_ADAM},
            {'params': self.episodic_memory.parameters(), 'lr': Config.LR_INNER_ADAM * 0.3}
        ])
        self.decoder_optimizer = optim.Adam([
            {'params': self.right_hemisphere.visual_extractor.decoder_proj.parameters(), 'lr': 1e-4},
            {'params': self.right_hemisphere.visual_extractor.decoder.parameters(), 'lr': 1e-4}
        ])        
        self.t_global = 0
        self.t_plasticity = 0
        self.last_loss_inner = 0.0
        self.last_curriculum_offset = Config.GOAL_MIN_OFFSET
        self.beta_schedule = 0
        
        self.to(device)
        self.update_target_networks(tau=1.0)

        self.last_td_loss = 0.0
        self.last_recon_loss = 0.0
        self.last_entropy_penalty = 0.0

        self.pathway_history = {'visual': [], 'semantic': []}


    def act(self, state, aux_features, epsilon=0.0, info=None):
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        
        with torch.no_grad():
            visual_out, audio_out, semantic_out, _, _, _ = self.right_hemisphere(state, aux_features, info=info)
            
            similar_episode, similarity = self.episodic_memory.retrieve_similar_episode(visual_out)
            
            if similar_episode is not None and similarity > Config.EPISODIC_MEMORY_THRESHOLD:
                episodic_boost = torch.ones(self.n_actions, dtype=torch.float32, device=self.device)
                
                if similar_episode in ['powerup', 'enemy_defeat', 'gap_jump']:
                    episodic_boost[2] *= 1.5
                    episodic_boost[4] *= 1.5
                elif similar_episode == 'pipe':
                    episodic_boost[1] *= 1.3
                elif similar_episode == 'secret':
                    episodic_boost[5] *= 1.4
            else:
                episodic_boost = torch.ones(self.n_actions, dtype=torch.float32, device=self.device)
            
            callosum_out, _, _ = self.corpus_callosum(visual_out, audio_out, semantic_out)
            q_vals = self.left_hemisphere(callosum_out)
            
            q_vals_boosted = q_vals * episodic_boost
            
            self.last_q_values = q_vals_boosted.cpu().numpy()
        
        return np.argmax(self.last_q_values)
    
    def remember(self, s, a, r, s_next, aux_s, aux_s_next, done, td_error=None):
        experience = {
            's': np.array(s, dtype=np.float32),
            'a': int(a),
            'r': float(r),
            's_next': np.array(s_next, dtype=np.float32),
            'aux_s': np.array(aux_s, dtype=np.float32),
            'aux_s_next': np.array(aux_s_next, dtype=np.float32),
            'done': bool(done)
        }
        self.memory.add(experience, td_error)
    
    def update_target_networks(self, tau=0.001):
        for target_param, param in zip(self.target_right_hemisphere.parameters(), self.right_hemisphere.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        
        for target_param, param in zip(self.target_corpus_callosum.parameters(), self.corpus_callosum.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        
        for target_param, param in zip(self.target_left_hemisphere.parameters(), self.left_hemisphere.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)



    def replay(self, batch_size=32, gamma=0.99):
        print(f"[DEBUG] Replay start - memory size: {len(self.memory)}")
        
        if len(self.memory) < batch_size:
            return 0.0
        
        actual_batch_size = min(batch_size, len(self.memory) // 2, 16)
        
        beta = min(1.0, Config.PRIORITY_BETA_START + self.beta_schedule * (1.0 - Config.PRIORITY_BETA_START) / Config.PRIORITY_BETA_FRAMES)
        self.beta_schedule += 1
        
        print(f"[DEBUG] Sampling batch...")
        batch, indices, weights = self.memory.sample(actual_batch_size, beta)
        
        print(f"[DEBUG] Creating tensors from batch...")
        if len(batch) == 0:
            return 0.0
        
        print(f"[DEBUG] Stacking states...")
        states = torch.stack([torch.from_numpy(b['s']) for b in batch]).to(self.device)
        print(f"[DEBUG] Stacking next_states...")
        next_states = torch.stack([torch.from_numpy(b['s_next']) for b in batch]).to(self.device)
        print(f"[DEBUG] Creating action tensor...")
        actions = torch.tensor([b['a'] for b in batch], dtype=torch.long, device=self.device)
        print(f"[DEBUG] Creating reward tensor...")
        rewards = torch.tensor([b['r'] for b in batch], dtype=torch.float32, device=self.device)
        print(f"[DEBUG] Creating dones tensor...")
        dones = torch.tensor([b['done'] for b in batch], dtype=torch.bool, device=self.device)
        print(f"[DEBUG] Stacking aux_features...")
        aux_features = torch.stack([torch.from_numpy(b['aux_s']) for b in batch]).to(self.device)
        print(f"[DEBUG] Stacking next_aux_features...")
        next_aux_features = torch.stack([torch.from_numpy(b['aux_s_next']) for b in batch]).to(self.device)
        print(f"[DEBUG] Creating weights tensor...")
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)
        
        print(f"[DEBUG] Computing surprise mask...")
        surprise_mask = (torch.abs(rewards) > Config.COIN_REWARD * 1.5).float()
        surprise_weights = 1.0 + surprise_mask * (Config.SURPRISE_LEARNING_RATE_MULTIPLIER - 1.0)
        
        print(f"[DEBUG] Computing target Q values...")
        with torch.no_grad():
            print(f"[DEBUG] Target right hemisphere forward...")
            visual_next, audio_next, semantic_next, _, _, _ = self.target_right_hemisphere(next_states, next_aux_features, info=None)
            print(f"[DEBUG] Target corpus callosum forward...")
            callosum_next, _, _ = self.target_corpus_callosum(visual_next, audio_next, semantic_next)
            print(f"[DEBUG] Target left hemisphere forward...")
            q_next = self.target_left_hemisphere(callosum_next)
            print(f"[DEBUG] Computing max Q...")
            max_q_next, _ = q_next.max(dim=-1)
            targets = rewards + Config.GAMMA_DISCOUNT * max_q_next * (~dones)
        
        print(f"[DEBUG] Computing current Q values...")
        visual, audio, semantic, visual_raw, aux_raw, visual_recon = self.right_hemisphere(states, aux_features, info=None)
        
        callosum_out, callosum_info, entropy_penalty = self.corpus_callosum(visual, audio, semantic)
        q_pred = self.left_hemisphere(callosum_out)
        if self.t_global % 100 == 0:
            with torch.no_grad():
                avg_memory_norm = self.left_hemisphere.memory_buffer.norm() / Config.LEFT_HEMISPHERE_MEMORY_WINDOW
                if avg_memory_norm < Config.GRADIENT_FLOW_THRESHOLD:
                    self.left_hemisphere.memory_buffer.mul_(1.1)
        q_pred_actions = q_pred.gather(1, actions.unsqueeze(1)).squeeze(1)
        td_errors = q_pred_actions - targets
        
        td_errors_clipped = torch.clamp(td_errors, -Config.PRIORITY_CLIP_TD_ERROR, Config.PRIORITY_CLIP_TD_ERROR)
        
        td_loss = (weights_tensor * surprise_weights * (td_errors_clipped ** 2)).mean()
        
        brain_loss = td_loss + entropy_penalty
        
        self.optimizer.zero_grad()
        brain_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.right_hemisphere.parameters()) + 
            list(self.corpus_callosum.parameters()) + 
            list(self.left_hemisphere.parameters()), 
            max_norm=1.0
        )
        self.optimizer.step()
        
        if self.t_global % Config.LIQUID_PLASTICITY_UPDATE_FREQ == 0:
            original_frames = states[:, -1, :, :]
            
            recon_has_nan = torch.isnan(visual_recon).any()
            frames_has_nan = torch.isnan(original_frames).any()
            
            if not recon_has_nan and not frames_has_nan:
                self.decoder_optimizer.zero_grad()
                
                recon_mse = nn.functional.mse_loss(visual_recon, original_frames)
                
                if torch.isfinite(recon_mse):
                    recon_mse.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.right_hemisphere.visual_extractor.decoder.parameters()) + 
                        list(self.right_hemisphere.visual_extractor.decoder_proj.parameters()), 
                        max_norm=0.5
                    )
                    self.decoder_optimizer.step()
                    self.last_recon_loss = float(recon_mse.detach())
                else:
                    self.last_recon_loss = 0.0
            else:
                self.last_recon_loss = 0.0
        else:
            self.last_recon_loss = 0.0
        
        self.t_plasticity += 1
        if self.t_plasticity % Config.LIQUID_PLASTICITY_UPDATE_FREQ == 0:
            batch_size_actual = len(batch)
            goal_emb_batch = torch.zeros(batch_size_actual, 1, dtype=torch.float32, device=self.device)
            aux_with_goal_batch = torch.cat([aux_raw[:, :self.right_hemisphere.aux_dim], goal_emb_batch], dim=1)
            
            td_abs_mean = torch.abs(td_errors_clipped).mean()
            
            plasticity_scale = torch.where(
                surprise_mask.mean() > 0.3,
                torch.tensor(Config.SURPRISE_LEARNING_RATE_MULTIPLIER, device=self.device),
                torch.tensor(1.0, device=self.device)
            )
            
            if td_abs_mean < 1.0:
                plasticity_scale = plasticity_scale * (1.0 + torch.clamp(td_abs_mean * 2.0, 0.0, 1.5))
            else:
                plasticity_scale = plasticity_scale * 0.5
            
            for i in range(batch_size_actual):
                td_err = td_errors_clipped[i].detach()
                surprise_boost = surprise_weights[i].item()
                
                plastic_grad_vis = self.right_hemisphere.visual_liquid.compute_plasticity_gradient(
                    visual_raw[i:i+1], visual[i:i+1], td_err
                )
                plastic_grad_sem = self.right_hemisphere.semantic_liquid.compute_plasticity_gradient(
                    aux_with_goal_batch[i:i+1], semantic[i:i+1], td_err
                )
                plastic_grad_audio = self.right_hemisphere.audio_liquid.compute_plasticity_gradient(
                    audio[i:i+1], audio[i:i+1], td_err
                )
                
                plastic_grad_vis = torch.nan_to_num(plastic_grad_vis, nan=0.0, posinf=0.1, neginf=-0.1)
                plastic_grad_sem = torch.nan_to_num(plastic_grad_sem, nan=0.0, posinf=0.1, neginf=-0.1)
                plastic_grad_audio = torch.nan_to_num(plastic_grad_audio, nan=0.0, posinf=0.1, neginf=-0.1)
                
                plastic_grad_vis *= plasticity_scale * surprise_boost
                plastic_grad_sem *= plasticity_scale * surprise_boost
                plastic_grad_audio *= plasticity_scale * surprise_boost * 1.5
                
                if self.right_hemisphere.visual_liquid.W_slow.grad is None:
                    self.right_hemisphere.visual_liquid.W_slow.grad = torch.zeros_like(self.right_hemisphere.visual_liquid.W_slow)
                if self.right_hemisphere.semantic_liquid.W_slow.grad is None:
                    self.right_hemisphere.semantic_liquid.W_slow.grad = torch.zeros_like(self.right_hemisphere.semantic_liquid.W_slow)
                if self.right_hemisphere.audio_liquid.W_slow.grad is None:
                    self.right_hemisphere.audio_liquid.W_slow.grad = torch.zeros_like(self.right_hemisphere.audio_liquid.W_slow)
                
                self.right_hemisphere.visual_liquid.W_slow.grad += plastic_grad_vis
                self.right_hemisphere.semantic_liquid.W_slow.grad += plastic_grad_sem
                self.right_hemisphere.audio_liquid.W_slow.grad += plastic_grad_audio
            
            self.right_hemisphere.visual_liquid.post_step_update()
            self.right_hemisphere.semantic_liquid.post_step_update()
            self.right_hemisphere.audio_liquid.post_step_update()
        
        self.memory.update_priorities(indices, td_errors.detach().cpu().numpy())
        
        self.t_global += 1
        
        if self.t_global % Config.TARGET_HARD_UPDATE_FREQUENCY == 0:
            self.update_target_networks(tau=1.0)
        elif self.t_global % Config.TARGET_UPDATE_FREQUENCY == 0:
            self.update_target_networks(tau=Config.TARGET_UPDATE_TAU)
        
        self.last_loss_inner = float(brain_loss)
        self.last_td_loss = float(td_loss)
        self.last_entropy_penalty = float(entropy_penalty)
        
        return float(brain_loss)
        


    def propose_goal(self, current_x, episode_num):
        MAX_LEVEL_X = 4000
        
        if episode_num < Config.CURRICULUM_WARMUP_EPISODES:
            offset = np.random.randint(Config.GOAL_MIN_OFFSET, Config.GOAL_MIN_OFFSET + 10)
        elif len(self.goal_history) >= Config.CURRICULUM_WINDOW:
            recent_goals = list(self.goal_history)[-Config.CURRICULUM_WINDOW:]
            recent_success = np.mean([s for _, s in recent_goals])
            
            if recent_success > 0.8:
                offset = min(Config.GOAL_MAX_OFFSET, self.last_curriculum_offset + Config.CURRICULUM_INCREMENT_SUCCESS)
            elif recent_success > 0.5:
                offset = self.last_curriculum_offset
            else:
                offset = max(Config.CURRICULUM_MIN_OFFSET, self.last_curriculum_offset - Config.CURRICULUM_DECREMENT_FAIL)
            
            self.last_curriculum_offset = offset
            offset += np.random.randint(-5, 6)
        else:
            offset = np.random.randint(Config.GOAL_MIN_OFFSET, Config.GOAL_MAX_OFFSET + 1)
        
        goal_x = min(current_x + offset, MAX_LEVEL_X)
        return goal_x
    
    def is_goal_achieved(self, goal_x, current_x):
        return current_x >= goal_x
    
    def reset(self):
        self.corpus_callosum.reset_fatigue()
        self.right_hemisphere.visual_liquid.homeostasis.fill_(1.0)
        self.right_hemisphere.visual_liquid.metabolism.fill_(0.6)
        self.right_hemisphere.visual_liquid.fatigue.fill_(0.0)
        self.right_hemisphere.semantic_liquid.homeostasis.fill_(1.0)
        self.right_hemisphere.semantic_liquid.metabolism.fill_(0.6)
        self.right_hemisphere.semantic_liquid.fatigue.fill_(0.0)
        self.right_hemisphere.audio_liquid.homeostasis.fill_(1.0)
        self.right_hemisphere.audio_liquid.metabolism.fill_(0.6)
        self.right_hemisphere.audio_liquid.fatigue.fill_(0.0)
        self.right_hemisphere.visual_liquid.eligibility_trace.zero_()
        self.right_hemisphere.semantic_liquid.eligibility_trace.zero_()
        self.right_hemisphere.audio_liquid.eligibility_trace.zero_()
        self.right_hemisphere.audio_generator.reset()
        
        self.left_hemisphere.reset_memory()
        
        self.prev_score_for_events.zero_()
        self.prev_coins_for_events.zero_()
        self.prev_y_pos_for_events.zero_()
        self.prev_x_pos_for_events.zero_()


def compute_focus_quality_score(saliency_map):
    """
    Calcula score de calidad del foco visual
    """
    coverage = np.sum(saliency_map > 0.1) / saliency_map.size
    
    if coverage > Config.SALIENCY_MIN_FOCUS_AREA:
        concentration = np.max(saliency_map) / (np.mean(saliency_map) + 1e-8)
    else:
        concentration = 0.0
    
    entropy = -np.sum(saliency_map * np.log(saliency_map + 1e-10))
    normalized_entropy = entropy / np.log(saliency_map.size)
    
    quality_score = (0.4 * (1 - coverage) + 
                     0.4 * min(concentration / 10.0, 1.0) + 
                     0.2 * (1 - normalized_entropy))
    
    return np.clip(quality_score, 0.0, 1.0)


    


env = gym.make("SuperMarioBros-1-1-v0")
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = FrameSkip(env, skip=Config.FRAMESKIP)
env = StuckMonitor(env, stuck_limit=Config.STUCK_LIMIT, inactivity_limit=Config.INACTIVITY_LIMIT)

print(f"✅ Acciones disponibles: {env.action_space.n}")
actions_names = ['NOOP', 'RIGHT', 'RIGHT+A', 'RIGHT+B', 'RIGHT+A+B', 'A', 'LEFT']
actions = actions_names[:env.action_space.n]

def get_aux_features(info, history=None):
    x_pos = info.get('x_pos', 0) / 256.0 
    y_pos = info.get('y_pos', 0) / 256.0 
    life = info.get('life', 2) / 3.0 
    score = info.get('score', 0) / 10000.0 
    time = info.get('time', 400) / 400.0 
    
    if history is not None and len(history) >= Config.TEMPORAL_HISTORY_WINDOW:
        recent_history = list(history)[-Config.TEMPORAL_HISTORY_WINDOW:]
        
        x_positions = [h.get('x_pos', 0) / 256.0 for h in recent_history]
        y_positions = [h.get('y_pos', 0) / 256.0 for h in recent_history]
        
        x_velocity = (x_positions[-1] - x_positions[0]) / len(x_positions)
        y_velocity = (y_positions[-1] - y_positions[0]) / len(y_positions)
        
        x_acceleration = 0.0
        y_acceleration = 0.0
        if len(x_positions) >= 3:
            v1_x = x_positions[1] - x_positions[0]
            v2_x = x_positions[-1] - x_positions[-2]
            x_acceleration = v2_x - v1_x
            
            v1_y = y_positions[1] - y_positions[0]
            v2_y = y_positions[-1] - y_positions[-2]
            y_acceleration = v2_y - v1_y
        
        is_airborne = float(abs(y_velocity) > 0.05)
        is_falling = float(y_velocity < -0.05)
        is_jumping = float(y_velocity > 0.05)
        movement_low = float(abs(x_velocity) < 0.02)
        
        return np.array([
            x_pos, y_pos, life, score, time,
            x_velocity, y_velocity, x_acceleration, y_acceleration,
            is_airborne, is_falling, is_jumping, movement_low
        ], dtype=np.float32)
    else:
        return np.array([
            x_pos, y_pos, life, score, time,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0
        ], dtype=np.float32)

agent = TricameralMarioAgent(n_actions=env.action_space.n, device=Config.DEVICE)
scores, losses, inner_losses, accuracies = [], [], [], []
epsilon, epsilon_min = 1.0, 0.05
start_episode = 0

EPISODES = 22

plt.ion()
fig, axs = plt.subplots(2, 2, figsize=(14, 9))
fig.patch.set_facecolor('#0a0a0a')
fig.suptitle('🧠 NeuroLogos TRICAMERAL V2.1 - Super Mario Bros (Plástica y Robusta)', 
             fontsize=16, color='#00ff00', weight='bold')
fig.tight_layout(pad=4.0)

for ax in axs.flat:
    ax.set_facecolor('#111')
    ax.grid(True, color='#333', alpha=0.3)
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#555')
    ax.spines['top'].set_color('#555')
    ax.spines['left'].set_color('#555')
    ax.spines['right'].set_color('#555')

print("🧠 NeuroLogos Tricameral V2.1 entrenando en Mario...")

for ep in tqdm(range(start_episode, EPISODES), initial=start_episode, total=EPISODES):
    obs, info = env.reset() 
    state = stack_frames(np.zeros((4,84,84), dtype=np.float32), obs, True)
    initial_info = info 
    aux_features = get_aux_features(initial_info) 
    
    total_reward, done, step = 0, False, 0
    ep_loss, loss_count = 0.0, 0
    
    current_x = initial_info.get('x_pos', 40)
    goal_x = agent.propose_goal(current_x, ep)
    goal_achieved = False
    last_life, last_score = initial_info.get('life', 2), initial_info.get('score', 0)
    last_coins = initial_info.get('coins', 0)
    
    agent.reset()
    power_up_gained = False
    info_history = deque(maxlen=Config.TEMPORAL_HISTORY_WINDOW)
    info_history.append(initial_info)    
    while not done:
        if len(agent.memory) < Config.MIN_MEMORY_BEFORE_LEARN:
            action = 1 
        else:
            action = agent.act(state, aux_features, epsilon, info=info)

        result = safe_step_with_timeout(env, action, timeout_seconds=5)
        if result[0] is None:  # Timeout detectado
            print("⚠️ Timeout - reiniciando episodio")
            break
        next_obs, game_reward, done, info = result
        info_history.append(info)
        next_aux_features = get_aux_features(info, history=info_history)
        env.render()

        current_x = info.get('x_pos', current_x)
        current_y = info.get('y_pos', agent.prev_y_pos_for_events.item())
        current_life = info.get('life', last_life)
        current_score = info.get('score', last_score)
        current_coins = info.get('coins', last_coins)
        next_aux_features = get_aux_features(info)
        aux_features = get_aux_features(info, history=info_history)
        score_diff = current_score - agent.prev_score_for_events.item()
        coin_diff = current_coins - agent.prev_coins_for_events.item()
        y_diff = current_y - agent.prev_y_pos_for_events.item()
        x_diff = current_x - agent.prev_x_pos_for_events.item()

        with torch.no_grad():
            state_t = torch.from_numpy(state).to(agent.device)
            aux_t = torch.from_numpy(aux_features).to(agent.device)
            visual_out, audio_out, semantic_out, _, _, _ = agent.right_hemisphere(state_t, aux_t, info=info)

        surprise_detected = False

        if score_diff > Config.POWER_UP_THRESHOLD and coin_diff == 0:
            agent.episodic_memory.store_episode('powerup', visual_out)
            surprise_detected = True
        elif score_diff >= 1000 and score_diff % 1000 == 0:
            agent.episodic_memory.store_episode('powerup', visual_out)
            surprise_detected = True

        if 50 <= score_diff <= 500 and score_diff not in [200, 1000]:
            agent.episodic_memory.store_episode('enemy_defeat', visual_out)
            surprise_detected = True

        if score_diff > 1000 and (score_diff % 1000 != 0):
            agent.episodic_memory.store_episode('secret', visual_out)
            surprise_detected = True

        if abs(y_diff) > 50 and x_diff > 20:
            agent.episodic_memory.store_episode('gap_jump', visual_out)
            surprise_detected = True

        if x_diff > 100 and abs(y_diff) < 10:
            agent.episodic_memory.store_episode('pipe', visual_out)
            surprise_detected = True

        agent.prev_score_for_events.copy_(torch.tensor(current_score))
        agent.prev_coins_for_events.copy_(torch.tensor(current_coins))
        agent.prev_y_pos_for_events.copy_(torch.tensor(current_y))
        agent.prev_x_pos_for_events.copy_(torch.tensor(current_x))

        current_x = info.get('x_pos', current_x)
        current_life = info.get('life', last_life)
        current_score = info.get('score', last_score)
        current_coins = info.get('coins', last_coins)
        next_aux_features = get_aux_features(info)

        if current_coins > last_coins:
            shaping_reward += Config.COIN_REWARD
        
        shaping_reward = 0.0

        if coin_diff > 0:
            shaping_reward += Config.COIN_REWARD
        
        power_up_gained = False
        if score_diff > 500 and coin_diff == 0:
            power_up_gained = True
        elif score_diff >= 1000 and score_diff % 1000 == 0:
            power_up_gained = True

        if power_up_gained:
            shaping_reward += Config.COIN_REWARD * 2.0
        elif score_diff > 0:
            shaping_reward += min(Config.MAX_SCORE_REWARD, score_diff * Config.SCORE_REWARD_SCALE)

        if not goal_achieved and agent.is_goal_achieved(goal_x, current_x):
            shaping_reward += Config.GOAL_REWARD
            goal_achieved = True

        death = (current_life < last_life)
        if death:
            shaping_reward += Config.DEATH_PENALTY

        stuck = info.get('stuck_steps', 0) >= Config.STUCK_LIMIT - 5
        inactive = info.get('inactive_steps', 0) >= Config.INACTIVITY_LIMIT - 5
        
        if stuck or inactive:
            shaping_reward += Config.STUCK_PENALTY

        if surprise_detected:
            shaping_reward *= Config.SURPRISE_LEARNING_RATE_MULTIPLIER

        final_reward = game_reward + shaping_reward

        
        stuck = info.get('stuck_steps', 0) >= Config.STUCK_LIMIT - 5
        inactive = info.get('inactive_steps', 0) >= Config.INACTIVITY_LIMIT - 5
        
        if stuck or inactive:
            final_reward += Config.STUCK_PENALTY

        next_state = stack_frames(state, next_obs, False)
        
        with torch.no_grad():
            state_t = torch.from_numpy(state).to(agent.device)
            aux_t = torch.from_numpy(aux_features).to(agent.device)
            next_state_t = torch.from_numpy(next_state).to(agent.device)
            next_aux_t = torch.from_numpy(next_aux_features).to(agent.device)
            
            visual, audio, semantic, visual_raw, aux_raw, _ = agent.right_hemisphere(state_t, aux_t)
            callosum_out, _, _ = agent.corpus_callosum(visual, audio, semantic)
            q_current = agent.left_hemisphere(callosum_out)
            
            visual_next, audio_next, semantic_next, _, _, _ = agent.target_right_hemisphere(next_state_t, next_aux_t)
            callosum_next, _, _ = agent.target_corpus_callosum(visual_next, audio_next, semantic_next)
            q_next = agent.target_left_hemisphere(callosum_next)
            
            q_current_action = q_current[action].item() if q_current.dim() > 0 else q_current.item()
            q_next_max = q_next.max().item() if q_next.dim() > 0 else q_next.item()
            
            if not np.isfinite(q_current_action):
                q_current_action = 0.0
            if not np.isfinite(q_next_max):
                q_next_max = 0.0
            
            td_error = (final_reward + 0.99 * q_next_max * (not done)) - q_current_action
            
            if not np.isfinite(td_error):
                print(f"[TD-ERROR-WARNING] td_error no finito: {td_error}, usando 0.0")
                td_error = 0.0
        
        agent.remember(state, action, final_reward, next_state, aux_features, next_aux_features, done, td_error)
        total_reward += final_reward

        if len(agent.memory) >= Config.MIN_MEMORY_BEFORE_LEARN:
            loss = agent.replay()
            ep_loss += loss
            loss_count += 1

        state = next_state.copy()
        aux_features = next_aux_features.copy()
        step += 1
        last_life, last_score, last_coins = current_life, current_score, current_coins

        if step % 20 == 0:
            for ax in axs.flat:
                ax.clear()
                ax.set_facecolor('#111')
                ax.grid(True, color='#333', alpha=0.3)
                ax.tick_params(colors='white')
            
            axs[0,0].imshow(state[0], cmap='viridis')
            axs[0,0].set_title("🎮 Frame Visual (Hemisferio Derecho)", 
                              color='#00ffff', fontsize=12, weight='bold', pad=10)
            axs[0,0].axis('off')
            axs[0,0].text(0.02, 0.98, f'Step: {step}\nMemoria: {len(agent.memory)}\nβ: {agent.beta_schedule/Config.PRIORITY_BETA_FRAMES:.3f}', 
                         transform=axs[0,0].transAxes, 
                         color='white', fontsize=9, 
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
            
            colors_q = ['#ff0000' if i == np.argmax(agent.last_q_values) else '#00ffff' 
                       for i in range(agent.n_actions)]
            
            q_values_clean = np.nan_to_num(agent.last_q_values, nan=0.0, posinf=1.0, neginf=-1.0)
            
            bars = axs[0,1].bar(range(agent.n_actions), q_values_clean, color=colors_q, edgecolor='white', linewidth=1.5)
            axs[0,1].set_title("⚡ Valores Q (Hemisferio Izquierdo)", 
                              color='#00ff00', fontsize=12, weight='bold', pad=10)
            axs[0,1].set_ylabel("Valor Q", color='white', fontsize=10)
            axs[0,1].set_xlabel("Acción", color='white', fontsize=10)
            axs[0,1].set_xticks(range(agent.n_actions))
            axs[0,1].set_xticklabels(actions_names[:agent.n_actions], rotation=45, ha='right', color='white', fontsize=8)
            
            y_min = min(-1.0, np.min(q_values_clean) - 0.5)
            y_max = max(1.0, np.max(q_values_clean) + 0.5)
            axs[0,1].set_ylim(y_min, y_max)
            
            for i, (bar, q) in enumerate(zip(bars, q_values_clean)):
                height = bar.get_height()
                label_y = height + 0.05 if height >= 0 else height - 0.05
                axs[0,1].text(bar.get_x() + bar.get_width()/2, label_y, 
                             f'{q:.2f}', ha='center', 
                             va='bottom' if height >= 0 else 'top',
                             color='white', fontsize=8, weight='bold')
            
            best_action = np.argmax(q_values_clean)
            axs[0,1].text(0.98, 0.98, f'Acción: {actions_names[best_action]}', 
                         transform=axs[0,1].transAxes, 
                         color='#ff0000', fontsize=10, weight='bold',
                         verticalalignment='top', horizontalalignment='right',
                         bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            
            h_vis = agent.right_hemisphere.visual_liquid.homeostasis
            h_sem = agent.right_hemisphere.semantic_liquid.homeostasis
            f_vis = agent.corpus_callosum.visual_fatigue
            f_sem = agent.corpus_callosum.semantic_fatigue
            
            trace_norm_vis = agent.right_hemisphere.visual_liquid.eligibility_trace.norm().item() / 100.0
            trace_norm_sem = agent.right_hemisphere.semantic_liquid.eligibility_trace.norm().item() / 100.0
            
            # Dentro del loop de visualización (if step % 20 == 0:)
            # Reemplazar la sección de Neurofisiología

            neuro_fisio = {
                'Visual Homeostasis': h_vis.item(),
                'Semantic Homeostasis': h_sem.item(),
                'Visual Fatiga': f_vis.item(),
                'Semantic Fatiga': f_sem.item(),
                'Trace Visual': trace_norm_vis,
                'Trace Semantic': trace_norm_sem,
                'Memory Filled': float(agent.left_hemisphere.memory_filled.item()),
                'LSTM Hidden Norm': agent.left_hemisphere.lstm_hidden.norm().item() / 10.0,
                r'Liquid LR ($\eta_L$)': Config.LR_LIQUID_PLASTICITY * 1e5
            }
            names = list(neuro_fisio.keys())
            values = list(neuro_fisio.values())
            colors = ['#00ffff', '#99ff99', '#ff3333', '#aa55ff', '#ffaa00', '#ff66cc', '#66ffaa']
            
            bars_neuro = axs[1,0].barh(names, values, color=colors[:len(names)], edgecolor='white', linewidth=1.5)
            axs[1,0].set_xlim(0, max(1.1, max(values)*1.1))
            axs[1,0].set_title("🧪 Neurofisiología Hemisferio Derecho (Plasticidad + Trace)", 
                              color='#ff00ff', fontsize=12, weight='bold', pad=10)
            axs[1,0].set_xlabel("Nivel de Activación/Fatiga", color='white', fontsize=10)
            
            for bar, val, name in zip(bars_neuro, values, names):
                width = bar.get_width()
                axs[1,0].text(width + 0.02, bar.get_y() + bar.get_height()/2,
                             f'{val:.3f}', ha='left', va='center',
                             color='white', fontsize=9, weight='bold')
            
            if len(agent.goal_history) > 0:
                acc = np.mean([s for _, s in agent.goal_history])
                if len(accuracies) == 0 or accuracies[-1] != acc:
                    accuracies.append(acc)
            else:
                if len(accuracies) == 0:
                    accuracies.append(0.0)
            
            inner_losses.append(agent.last_loss_inner)
            
            window = min(100, max(len(scores), len(losses), len(accuracies)))
            
            if len(scores) > 0:
                scores_window = scores[-window:]
                scores_clean = [s if np.isfinite(s) else 0.0 for s in scores_window]
                
                if any(s != 0.0 for s in scores_clean):
                    x_range = range(len(scores_clean))
                    axs[1,1].plot(x_range, scores_clean, color='#00ff00', linewidth=2, 
                                 label='Recompensa Promedio (TD)', alpha=0.9, marker='o', markersize=2)
            
            if len(losses) > 0:
                losses_window = losses[-window:]
                losses_clean = [l if np.isfinite(l) and l > 0 else 1e-6 for l in losses_window]
                
                if len(losses_clean) > 0 and max(losses_clean) > 0:
                    x_range = range(len(losses_clean))
                    loss_scale = max(1e-3, np.percentile(losses_clean, 90))
                    scaled_losses = [l/loss_scale for l in losses_clean]
                    axs[1,1].plot(x_range, scaled_losses, color='#ff3333', linewidth=2, 
                                 label=f'Pérdida Total (÷{loss_scale:.2e})', alpha=0.9, linestyle='--')
            
            if len(inner_losses) > 0:
                inner_losses_window = inner_losses[-window:]
                inner_clean = [l if np.isfinite(l) and l > 0 else 1e-8 for l in inner_losses_window]
                
                if len(inner_clean) > 0 and max(inner_clean) > 0:
                    x_range = range(len(inner_clean))
                    inner_loss_scale = max(1e-5, np.percentile(inner_clean, 90))
                    scaled_inner = [l/inner_loss_scale for l in inner_clean]
                    axs[1,1].plot(x_range, scaled_inner, color='#ffaa00', linewidth=1.5, 
                                 label=f'Pérdida Interna (÷{inner_loss_scale:.2e})', alpha=0.7, linestyle=':')
            
            if len(accuracies) > 0:
                acc_window = accuracies[-window:]
                acc_clean = [a if np.isfinite(a) else 0.0 for a in acc_window]
                x_range = range(len(acc_clean))
                axs[1,1].plot(x_range, acc_clean, color='#99ccff', linewidth=2.5, 
                             label='Precisión Metas', alpha=0.9, marker='s', markersize=2)
            
            if not hasattr(agent, 'pathway_history'):
                agent.pathway_history = {'visual': [], 'semantic': []}
            
            if len(agent.corpus_callosum.pathway_grad_norms) > 0:
                vis_grad = agent.corpus_callosum.pathway_grad_norms[0].item()
                sem_grad = agent.corpus_callosum.pathway_grad_norms[2].item()
                total = vis_grad + sem_grad + 1e-10
                
                agent.pathway_history['visual'].append(vis_grad / total)
                agent.pathway_history['semantic'].append(sem_grad / total)
            
            if len(agent.pathway_history['visual']) > 0:
                pathway_window = min(100, len(agent.pathway_history['visual']))
                
                vis_data = agent.pathway_history['visual'][-pathway_window:]
                sem_data = agent.pathway_history['semantic'][-pathway_window:]
                
                vis_data_clean = [v if np.isfinite(v) else 0.0 for v in vis_data]
                sem_data_clean = [s if np.isfinite(s) else 0.0 for s in sem_data]
                
                x_range = range(len(vis_data_clean))
                
                axs[1,1].plot(x_range, vis_data_clean, 
                             color='#ff00ff', linewidth=2, label='Visual Contrib', alpha=0.8, linestyle='-.')
                axs[1,1].plot(x_range, sem_data_clean, 
                             color='#ffff00', linewidth=2, label='Semantic Contrib', alpha=0.8, linestyle='-.')
            
            axs[1,1].set_title(f"📈 Entrenamiento | Ep: {ep} | Meta: X≥{goal_x} | ε={epsilon:.3f} | Offset: {agent.last_curriculum_offset}", 
                              color='#00ffff', fontsize=11, weight='bold', pad=10)
            axs[1,1].set_xlabel("Episodios Recientes", color='white', fontsize=10)
            axs[1,1].set_ylabel("Valor Normalizado", color='white', fontsize=10)
            axs[1,1].legend(loc='upper left', fontsize=8, 
                          facecolor='#222', labelcolor='white', 
                          framealpha=0.9, edgecolor='white')
            axs[1,1].axhline(y=0, color='white', linestyle='-', alpha=0.2)
            axs[1,1].grid(True, color='#333', alpha=0.3)
            
            recent_scores = scores[-10:] if len(scores) >= 10 else scores
            recent_scores_clean = [s for s in recent_scores if np.isfinite(s)]
            
            avg_reward = np.mean(recent_scores_clean) if recent_scores_clean else 0.0
            avg_loss = ep_loss / max(loss_count, 1)
            
            goal_success = np.mean([s for _, s in agent.goal_history]) if len(agent.goal_history) > 0 else 0.0
            
            stats_text = f'Avg Reward (10): {avg_reward:.2f}\n'
            stats_text += f'Loss TD: {avg_loss:.4f}\n'
            stats_text += f'Loss Interna: {agent.last_loss_inner:.5f}\n'
            stats_text += f'Goal Success: {goal_success*100:.1f}%\n'
            stats_text += f'X Position: {current_x}\n'
            stats_text += f'Coins: {current_coins} | Score: {current_score}\n'
            stats_text += f'Curriculum Offset: {agent.last_curriculum_offset}'
            
            axs[1,1].text(0.98, 0.02, stats_text,
                         transform=axs[1,1].transAxes,
                         color='white', fontsize=8,
                         verticalalignment='bottom', horizontalalignment='right',
                         bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
                         family='monospace')
            
            plt.pause(0.01)

    success = agent.is_goal_achieved(goal_x, current_x)
    agent.goal_history.append((goal_x, success))
    scores.append(total_reward if np.isfinite(total_reward) else 0.0)
    avg_loss = ep_loss / max(loss_count, 1)
    losses.append(avg_loss if np.isfinite(avg_loss) else 1e-6)

    # Plotting seguro
    window = min(100, max(len(scores), len(losses), len(accuracies)))

    if len(scores) > 0:
        scores_window = scores[-window:]
        scores_clean = [s if np.isfinite(s) else 0.0 for s in scores_window]
        
        if any(s != 0.0 for s in scores_clean):
            x_range = range(len(scores_clean))
            axs[1,1].plot(x_range, scores_clean, color='#00ff00', linewidth=2, 
                        label='Recompensa Promedio (TD)', alpha=0.9, marker='o', markersize=2)

    epsilon = max(epsilon_min, epsilon * Config.EXPLORATION_DECAY_RATE)

    if (ep + 1) % Config.CHECKPOINT_EVERY == 0:
        curriculum_info = f"Offset={agent.last_curriculum_offset}"
        
        acc_value = accuracies[-1] * 100 if len(accuracies) > 0 else 0.0
        avg_reward = np.mean(scores[-10:]) if len(scores) >= 10 else (np.mean(scores) if len(scores) > 0 else 0.0)
        
        print(f"\n💾 Checkpoint ep {ep+1} | ε={epsilon:.3f} | Acc={acc_value:.1f}% | Avg Reward={avg_reward:.2f} | {curriculum_info}")

env.close()
plt.ioff()
plt.show()
print("✅ Entrenamiento Tricameral finalizado.")