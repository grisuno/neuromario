#!/usr/bin/env python3
# _*_ coding: utf8 _*_
"""
app.py

Autor: Gris Iscomeback
Correo electrónico: grisiscomeback[at]gmail[dot]com
Fecha de creación: xx/xx/xxxx
Licencia: GPL v3

Descripción:  
"""
# --- visualizacion_foco_real.py ---

import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2 

import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

try:
    from umap import UMAP
    print("✅ UMAP cargado para visualización 3D.")
except ImportError:
    print("⚠️ Instala umap-learn para ver el mapa 3D.")

# ************************************************************
# 🟢 IMPORTACIÓN DESDE TU ARCHIVO 🟢
# ************************************************************

try:
    from trimario4 import (
        Config, FrameSkip, StuckMonitor, 
        TricameralMarioAgent, 
        RightHemisphere, LeftHemisphere, CorpusCallosum, 
        stack_frames, get_aux_features
    )
    print("✅ Módulos cargados correctamente.")
except ImportError as e:
    print(f"🚨 Error de importación: {e}")
    exit(1)

# ************************************************************
# LÓGICA DE CÁLCULO DE SALIENCY (FOCO REAL)
# ************************************************************

def compute_real_saliency(agent, state, aux_features):
    agent.eval()
    
    input_tensor = torch.from_numpy(state).unsqueeze(0).to(Config.DEVICE)
    input_tensor.requires_grad = True
    
    aux_tensor = torch.from_numpy(aux_features).unsqueeze(0).to(Config.DEVICE) if isinstance(aux_features, np.ndarray) else aux_features
    
    with torch.enable_grad():
        visual_out, audio_sim, semantic_out, _, _, _ = agent.right_hemisphere(
            input_tensor, aux_tensor, visualization_mode=True
        )
        callosum_out, callosum_info, _ = agent.corpus_callosum(visual_out, audio_sim, semantic_out)
        q_vals = agent.left_hemisphere(callosum_out)
        
        score = q_vals.max()
        
        if input_tensor.grad is not None:
            input_tensor.grad.zero_()
        
        score.backward(retain_graph=False)
    
    if input_tensor.grad is None:
        print("⚠️ No se pudo calcular gradientes - usando mapa de atención uniforme")
        saliency_final = np.ones((84, 84), dtype=np.float32) * 0.5
        focus_metrics = {
            'total_coverage': 0.5,
            'peak_intensity': 0.5,
            'mean_intensity': 0.5,
            'focus_entropy': 1.0
        }
        return saliency_final, focus_metrics
    
    saliency, _ = torch.max(torch.abs(input_tensor.grad[0]), dim=0)
    saliency = saliency.cpu().numpy()
    
    # Solución al problema de HUD
    hud_mask = np.zeros((84, 84), dtype=np.float32)
    hud_mask[0:20, :] = 1.0  # Barra superior (score, coins, time)
    hud_mask[-15:, :] = 1.0  # Barra inferior (world, stage)
    
    saliency_normalized = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    
    saliency_sharpened = np.power(saliency_normalized, Config.SALIENCY_SHARPENING_POWER)
    
    # Aplicar HUD mask antes del thresholding
    saliency_with_hud = saliency_sharpened + hud_mask * 0.3
    
    threshold = np.percentile(saliency_with_hud, Config.SALIENCY_PERCENTILE_THRESHOLD)
    saliency_masked = np.where(saliency_with_hud > threshold, saliency_with_hud, 0.0)
    
    top_k_threshold = np.percentile(saliency_masked[saliency_masked > 0], 80) if np.any(saliency_masked > 0) else 0
    saliency_sparse = np.where(saliency_masked > top_k_threshold, saliency_masked, saliency_masked * 0.1)
    
    saliency_blurred = cv2.GaussianBlur(saliency_sparse, (Config.SALIENCY_BLUR_KERNEL, Config.SALIENCY_BLUR_KERNEL), 0)
    
    saliency_final = (saliency_blurred - saliency_blurred.min()) / (saliency_blurred.max() - saliency_blurred.min() + 1e-8)
    
    focus_metrics = {
        'total_coverage': np.sum(saliency_final > 0.1) / saliency_final.size,
        'peak_intensity': np.max(saliency_final),
        'mean_intensity': np.mean(saliency_final[saliency_final > 0]) if np.any(saliency_final > 0) else 0.0,
        'focus_entropy': -np.sum(saliency_final * np.log(saliency_final + 1e-10))
    }
    
    # Penalización de no-foco semántico:
    if focus_metrics['total_coverage'] < 0.1:
        focus_metrics['focus_entropy'] += 0.5
    
    return saliency_final, focus_metrics

# ************************************************************
# RECOLECCIÓN DE DATOS
# ************************************************************

def run_and_log_activations(agent, env, num_steps=300):
    obs, info = env.reset() 
    state = stack_frames(np.zeros((4,84,84), dtype=np.float32), obs, True)
    aux_features = get_aux_features(info)
    
    log_data = {
        'callosum_fused': [],      
        'visual_liquid': [],       
        'semantic_liquid': [],     
        'left_executive_a2': [],   
        'context_weights': [],     
        'input_frames': [],
        'q_values': [],
        'actions': [],
        'td_errors': [],
        'pathway_contributions': {'visual': [], 'semantic': []},
        'real_saliency': None,
        'focus_metrics': None,
        'cross_modal_coherence': []
    }
    
    current_x = info.get('x_pos', 40)
    goal_x = agent.propose_goal(current_x, 0)
    
    print(f"\n--- Recolectando {num_steps} pasos ---")
    
    for step in tqdm(range(num_steps)):
        agent.eval()
        
        log_data['input_frames'].append(state.copy())
        
        with torch.no_grad():
            v_out, a_sim, s_out, _, _, _ = agent.right_hemisphere(state, aux_features, goal_x, current_x)
            c_out, c_info, _ = agent.corpus_callosum(v_out, a_sim, s_out)
            q_vals, cache_left = agent.left_hemisphere.forward_with_cache(c_out)
        
        action = np.argmax(q_vals.cpu().numpy())
        
        log_data['callosum_fused'].append(c_out.cpu().numpy().flatten())
        log_data['visual_liquid'].append(v_out.cpu().numpy().flatten())
        log_data['semantic_liquid'].append(s_out.cpu().numpy().flatten())
        log_data['left_executive_a2'].append(cache_left['a2'].cpu().numpy().flatten())
        log_data['context_weights'].append(c_info['context_weights'].cpu().numpy().flatten())
        log_data['q_values'].append(q_vals.cpu().numpy())
        log_data['actions'].append(action)
        
        vis_grad = agent.corpus_callosum.pathway_grad_norms[0].item()
        sem_grad = agent.corpus_callosum.pathway_grad_norms[2].item()
        total_grad = vis_grad + sem_grad + 1e-10
        log_data['pathway_contributions']['visual'].append(vis_grad / total_grad)
        log_data['pathway_contributions']['semantic'].append(sem_grad / total_grad)
        
        cross_modal_coherence = torch.cosine_similarity(
            v_out.flatten().unsqueeze(0), 
            s_out.flatten().unsqueeze(0)
        ).item()
        log_data['cross_modal_coherence'].append(cross_modal_coherence)
        
        next_obs, _, done, info = env.step(action)
        state = stack_frames(state, next_obs, False)
        aux_features = get_aux_features(info)
        current_x = info.get('x_pos', current_x)
        
        if done: 
            break
            
    print("\n🔎 Calculando foco visual real con sharpening...")
    log_data['real_saliency'], log_data['focus_metrics'] = compute_real_saliency(agent, state, aux_features)
    
    for key in log_data:
        if key not in ['real_saliency', 'focus_metrics', 'pathway_contributions', 'cross_modal_coherence']:
            log_data[key] = np.array(log_data[key])
    
    log_data['pathway_contributions']['visual'] = np.array(log_data['pathway_contributions']['visual'])
    log_data['pathway_contributions']['semantic'] = np.array(log_data['pathway_contributions']['semantic'])
    log_data['cross_modal_coherence'] = np.array(log_data['cross_modal_coherence'])
            
    return log_data



# ************************************************************
# VISUALIZACIÓN UNIFICADA
# ************************************************************

def plot_activation_3d_and_bars(log_data):
    callosum_outputs = log_data['callosum_fused']
    reducer = UMAP(n_components=3, random_state=42)
    embedding = reducer.fit_transform(callosum_outputs)
    
    avg_activations = {
        'Visual Liquid (H.D.)': np.mean(np.linalg.norm(log_data['visual_liquid'], axis=1)),
        'Semantic Liquid (H.D.)': np.mean(np.linalg.norm(log_data['semantic_liquid'], axis=1)),
        'Executive Layer (H.I.)': np.mean(np.linalg.norm(log_data['left_executive_a2'], axis=1)),
    }
    
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor('#05051a')
    plt.suptitle('🧠 Cerebro Tricameral: Análisis de Foco Real y Activación', color='#00FF00', fontsize=18)
    
    ax_3d = fig.add_subplot(3, 3, (1, 3), projection='3d')
    ax_3d.set_facecolor('#05051a')
    ax_3d.plot(embedding[:, 0], embedding[:, 1], embedding[:, 2], color='#00FFFF', alpha=0.4)
    scatter = ax_3d.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], 
                           c=np.arange(len(embedding)), cmap='plasma', s=30)
    ax_3d.set_title('Trayectoria del Estado Fusionado', color='white')
    plt.colorbar(scatter, ax=ax_3d, label='Paso Temporal')
    
    ax_flow = fig.add_subplot(3, 3, 4)
    avg_weights = np.mean(log_data['context_weights'], axis=0)
    bars = ax_flow.bar(['Visual', 'Audio/Sim.', 'Semántico'], avg_weights, 
                       color=['#00FF00', '#FF00FF', '#00FFFF'])
    ax_flow.set_title('Prioridad de Caminos (Callosum)', color='white')
    ax_flow.set_facecolor('#1a1a2e')
    ax_flow.set_ylabel('Peso Promedio', color='white')
    for bar, val in zip(bars, avg_weights):
        height = bar.get_height()
        ax_flow.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', color='white', fontsize=9)
    
    ax_act = fig.add_subplot(3, 3, 5)
    bars_act = ax_act.bar(avg_activations.keys(), avg_activations.values(), 
                          color=['#00FF00', '#FF00FF', '#FFFFFF'])
    ax_act.set_title('Intensidad de Activación Modular', color='white')
    ax_act.set_facecolor('#1a1a2e')
    ax_act.set_ylabel('Norma L2 Promedio', color='white')
    plt.xticks(rotation=15, color='white')
    for bar, val in zip(bars_act, avg_activations.values()):
        height = bar.get_height()
        ax_act.text(bar.get_x() + bar.get_width()/2, height + 1,
                   f'{val:.1f}', ha='center', va='bottom', color='white', fontsize=9)
    
    ax_saliency = fig.add_subplot(3, 3, 6)
    last_frame = log_data['input_frames'][-1][-1, :, :]
    saliency_map = log_data['real_saliency']
    
    ax_saliency.imshow(last_frame, cmap='gray')
    overlay = ax_saliency.imshow(saliency_map, cmap='hot', alpha=0.7)
    ax_saliency.set_title('FOCO REAL (Áreas de Interés Neuronal)', color='#FF00FF', fontsize=11)
    ax_saliency.axis('off')
    plt.colorbar(overlay, ax=ax_saliency, label='Intensidad de Atención')
    
    focus_metrics = log_data['focus_metrics']
    metrics_text = f"Cobertura: {focus_metrics['total_coverage']*100:.1f}%\n"
    metrics_text += f"Pico: {focus_metrics['peak_intensity']:.3f}\n"
    metrics_text += f"Media: {focus_metrics['mean_intensity']:.3f}\n"
    metrics_text += f"Entropía: {focus_metrics['focus_entropy']:.2f}"
    ax_saliency.text(0.02, 0.98, metrics_text, transform=ax_saliency.transAxes,
                    color='white', fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    ax_pathway = fig.add_subplot(3, 3, 7)
    steps = np.arange(len(log_data['pathway_contributions']['visual']))
    ax_pathway.plot(steps, log_data['pathway_contributions']['visual'], 
                   color='#00FF00', linewidth=2, label='Visual', alpha=0.8)
    ax_pathway.plot(steps, log_data['pathway_contributions']['semantic'], 
                   color='#FF00FF', linewidth=2, label='Semántico', alpha=0.8)
    ax_pathway.set_title('Contribución de Pathways en el Tiempo', color='white')
    ax_pathway.set_facecolor('#1a1a2e')
    ax_pathway.set_xlabel('Paso', color='white')
    ax_pathway.set_ylabel('Contribución Normalizada', color='white')
    ax_pathway.legend(loc='upper right', facecolor='#222', labelcolor='white')
    ax_pathway.grid(True, color='#333', alpha=0.3)
    
    ax_coherence = fig.add_subplot(3, 3, 8)
    ax_coherence.plot(steps, log_data['cross_modal_coherence'], 
                     color='#00FFFF', linewidth=2, alpha=0.8)
    ax_coherence.set_title('Coherencia Cross-Modal (Visual ↔ Semántico)', color='white')
    ax_coherence.set_facecolor('#1a1a2e')
    ax_coherence.set_xlabel('Paso', color='white')
    ax_coherence.set_ylabel('Similitud de Coseno', color='white')
    ax_coherence.axhline(y=0, color='white', linestyle='--', alpha=0.3)
    ax_coherence.grid(True, color='#333', alpha=0.3)
    
    ax_qvals = fig.add_subplot(3, 3, 9)
    q_values = log_data['q_values']
    max_q_per_step = np.max(q_values, axis=1)
    actions = log_data['actions']
    
    ax_qvals.plot(steps, max_q_per_step, color='#FFD700', linewidth=2, alpha=0.8, label='Max Q-Value')
    ax_qvals.scatter(steps, max_q_per_step, c=actions, cmap='tab10', s=20, alpha=0.6)
    ax_qvals.set_title('Evolución de Q-Values y Acciones', color='white')
    ax_qvals.set_facecolor('#1a1a2e')
    ax_qvals.set_xlabel('Paso', color='white')
    ax_qvals.set_ylabel('Q-Value Máximo', color='white')
    ax_qvals.legend(loc='upper left', facecolor='#222', labelcolor='white')
    ax_qvals.grid(True, color='#333', alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()



def visualize_attention_layers(agent, state, aux_features):
    """
    Visualiza mapas de atención en diferentes etapas del procesamiento
    """
    agent.eval()
    
    input_tensor = torch.from_numpy(state).unsqueeze(0).to(Config.DEVICE)
    input_tensor.requires_grad = True
    
    aux_tensor = torch.from_numpy(aux_features).unsqueeze(0).to(Config.DEVICE) if isinstance(aux_features, np.ndarray) else aux_features
    
    attention_maps = {}
    
    for action_idx in range(agent.n_actions):
        if input_tensor.grad is not None:
            input_tensor.grad.zero_()
        
        visual_out, audio_sim, semantic_out, _, _, _ = agent.right_hemisphere(
            input_tensor, aux_tensor, visualization_mode=True
        )
        callosum_out, callosum_info, _ = agent.corpus_callosum(visual_out, audio_sim, semantic_out)
        q_vals = agent.left_hemisphere(callosum_out)
        
        if q_vals.dim() == 1:
            q_vals = q_vals.unsqueeze(0)
        
        q_vals[0, action_idx].backward(retain_graph=(action_idx < agent.n_actions - 1))
        
        if input_tensor.grad is not None:
            saliency = torch.max(torch.abs(input_tensor.grad[0]), dim=0)[0]
            saliency_np = saliency.cpu().numpy()
            saliency_np = (saliency_np - saliency_np.min()) / (saliency_np.max() - saliency_np.min() + 1e-8)
            attention_maps[action_idx] = saliency_np
        else:
            attention_maps[action_idx] = np.ones((84, 84), dtype=np.float32) * 0.5
    
    return attention_maps



if __name__ == "__main__":
    env = JoypadSpace(gym.make("SuperMarioBros-1-1-v0"), SIMPLE_MOVEMENT)
    env = FrameSkip(env, skip=Config.FRAMESKIP)
    env = StuckMonitor(env, stuck_limit=Config.STUCK_LIMIT, inactivity_limit=Config.INACTIVITY_LIMIT)
    
    agent = TricameralMarioAgent(n_actions=env.action_space.n, device=Config.DEVICE)
    agent.reset()

    log_data = run_and_log_activations(agent, env, num_steps=300)
    plot_activation_3d_and_bars(log_data)

    obs, info = env.reset()
    state = stack_frames(np.zeros((4,84,84), dtype=np.float32), obs, True)
    aux_features = get_aux_features(info)
    
    attention_per_action = visualize_attention_layers(agent, state, aux_features)
    
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Mapas de Atención por Acción', color='white', fontsize=14)
    actions_names = ['NOOP', 'RIGHT', 'RIGHT+A', 'RIGHT+B', 'RIGHT+A+B', 'A', 'LEFT']
    
    for idx, ax in enumerate(axes.flat):
        if idx < agent.n_actions:
            ax.imshow(state[-1], cmap='gray')
            ax.imshow(attention_per_action[idx], cmap='hot', alpha=0.6)
            ax.set_title(actions_names[idx], color='white')
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()
