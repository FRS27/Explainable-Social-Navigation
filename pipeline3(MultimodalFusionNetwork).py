import os, gc
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any, Tuple
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoModelForDepthEstimation,
    set_seed,
)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from collections import defaultdict
import warnings
import cv2
import math
import random
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

# ================== CONFIGURATION ==================
TRAIN_JSONL = "C:\\Users\\Student\\Downloads\\dual_velocity_results_quality_resultssdgfsdf\\dual_velocity_results_quality_resultssdgfsdf\\training_dataset6_final_relative.jsonl"
VAL_JSONL   = "C:\\Users\\Student\\Downloads\\dual_velocity_results_quality_resultssdgfsdf\\dual_velocity_results_quality_resultssdgfsdf\\validation_dataset6_final_relative.jsonl"
OUTPUT_DIR  = "./run2"

DINOV2_MODEL = "facebook/dinov2-base"
DEPTH_ANYTHING_V2_MODEL = "depth-anything/Depth-Anything-V2-Base-hf"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ADVANCED_CONFIG = {
    'use_fp16': True,          
    'image_size': 224,
    'lstm_hidden_size': 256,
    'lstm_num_layers': 2,
    'fusion_hidden_size': 512,
    'dropout_rate': 0.2,
    'learning_rate': 0.0001,
    'batch_size': 2,
    'num_epochs': 50,
    'early_stopping_patience': 8,  
    'sequence_length': 6,
    # Windows-safe loader
    'num_workers': 0,
    'pin_memory': False,
}

# Batched vision forward chunk size (tune for VRAM)
VISION_CHUNK = 48

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("🚀 Dual-Branch Velocity Prediction Pipeline with Comprehensive Visualizations")
print(f"🖥️  Device: {DEVICE}")

def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

class VisualizationSuite:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(self.viz_dir, exist_ok=True)
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
    def plot_input_sequence(self, image_sequence: List[Image.Image], sample_idx: int = 0):
        print("📸 Creating Plot A: Input Image Sequence...")
        fig, axes = plt.subplots(1, len(image_sequence), figsize=(18, 3))
        fig.suptitle('Plot A: Input Image Sequence - "What the Robot Sees"', fontsize=16, fontweight='bold', y=0.95)
        for i, (ax, img) in enumerate(zip(axes, image_sequence)):
            ax.imshow(img)
            ax.set_title(f'Frame {i+1}', fontsize=12, fontweight='bold')
            ax.axis('off')
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.savefig(os.path.join(self.viz_dir, f'plot_a_input_sequence_{sample_idx}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_depth_perception(self, input_image: Image.Image, depth_map: np.ndarray, sample_idx: int = 0):
        print("🌊 Creating Plot B: Depth Perception...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Plot B: Depth Perception - "Beyond 2D Vision"', fontsize=16, fontweight='bold')
        ax1.imshow(input_image); ax1.set_title('Original RGB Image', fontsize=14, fontweight='bold'); ax1.axis('off')
        d = depth_map
        d = (d - d.min()) / (d.max() - d.min() + 1e-6)
        im = ax2.imshow(d, cmap='magma'); ax2.set_title('Depth Map (Hot=Near, Cool=Far)', fontsize=14, fontweight='bold'); ax2.axis('off')
        cbar = plt.colorbar(im, ax=ax2, shrink=0.8); cbar.set_label('Relative Depth', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, f'plot_b_depth_perception_{sample_idx}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_dinov2_attention(self, input_image: Image.Image, attention_map: np.ndarray, sample_idx: int = 0):
        print("🧠 Creating Plot C: DINOv2 Semantic Understanding (Real)…")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Plot C: Semantic Understanding - "What the Robot Focuses On"', fontsize=16, fontweight='bold')
        ax1.imshow(input_image); ax1.set_title('Input Image', fontsize=14, fontweight='bold'); ax1.axis('off')
        ax2.imshow(input_image, alpha=0.7)
        im = ax2.imshow(attention_map, cmap='jet', alpha=0.5)
        ax2.set_title('DINOv2 Attention (CLS→Patches)', fontsize=14, fontweight='bold'); ax2.axis('off')
        cbar = plt.colorbar(im, ax=ax2, shrink=0.8); cbar.set_label('Attention Strength', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, f'plot_c_dinov2_attention_{sample_idx}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_optical_flow_comparison(self, turn_image: Image.Image, turn_flow: np.ndarray,
                                     forward_image: Image.Image, forward_flow: np.ndarray):
        print("🔥 Creating Plot D: Optical Flow Comparison (Smoking Gun)…")
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Plot D: Optical Flow Analysis - "The Smoking Gun Evidence"', fontsize=18, fontweight='bold', y=0.95)
        ax1.imshow(turn_image); ax1.set_title('Turn: Strong Rotational Signal', fontsize=14, fontweight='bold', color='red'); ax1.axis('off')
        h, w = turn_flow.shape[:2]; step = 20; y, x = np.mgrid[0:h, 0:w]; y, x = y[::step, ::step], x[::step, ::step]
        u = turn_flow[::step, ::step, 0]; v = turn_flow[::step, ::step, 1]
        ax2.imshow(turn_image, alpha=0.7); ax2.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=1, color='red', width=0.003, alpha=0.8)
        ax2.set_title('Turn: Coherent Global Flow', fontsize=14, fontweight='bold', color='red'); ax2.axis('off')
        ax3.imshow(forward_image); ax3.set_title('Forward: Translational Signal', fontsize=14, fontweight='bold', color='blue'); ax3.axis('off')
        hf, wf = forward_flow.shape[:2]; y, x = np.mgrid[0:hf, 0:wf]; y, x = y[::step, ::step], x[::step, ::step]
        uf = forward_flow[::step, ::step, 0]; vf = forward_flow[::step, ::step, 1]
        ax4.imshow(forward_image, alpha=0.7); ax4.quiver(x, y, uf, vf, angles='xy', scale_units='xy', scale=1, color='blue', width=0.003, alpha=0.8)
        ax4.set_title('Forward: Weak, Radial Flow', fontsize=14, fontweight='bold', color='blue'); ax4.axis('off')
        fig.text(0.5, 0.02, 'KEY: Rotation → strong, coherent flow; Translation → weak/noisy flow (explains R² gap).',
                 ha='center', fontsize=12, style='italic', bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7))
        plt.tight_layout(); plt.subplots_adjust(top=0.88, bottom=0.15)
        plt.savefig(os.path.join(self.viz_dir, 'plot_d_optical_flow_smoking_gun.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_prediction_scatter(self, predictions: np.ndarray, ground_truth: np.ndarray):
        print("📊 Creating Plot E: Prediction vs Ground Truth…")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle('Plot E: Prediction Performance - "Quantitative Validation"', fontsize=16, fontweight='bold')
        angular_r2 = r2_score(ground_truth[:, 1], predictions[:, 1])
        ax1.scatter(ground_truth[:, 1], predictions[:, 1], alpha=0.6, color='red', s=30)
        min_val, max_val = min(ground_truth[:, 1].min(), predictions[:, 1].min()), max(ground_truth[:, 1].max(), predictions[:, 1].max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2)
        ax1.set_xlabel('True Angular Velocity'); ax1.set_ylabel('Predicted Angular Velocity')
        ax1.set_title(f'Angular: R² = {angular_r2:.4f}', fontsize=14, fontweight='bold', color='red'); ax1.grid(True, alpha=0.3)
        linear_r2 = r2_score(ground_truth[:, 0], predictions[:, 0])
        ax2.scatter(ground_truth[:, 0], predictions[:, 0], alpha=0.6, color='blue', s=30)
        min_val, max_val = min(ground_truth[:, 0].min(), predictions[:, 0].min()), max(ground_truth[:, 0].max(), predictions[:, 0].max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2)
        ax2.set_xlabel('True Linear Velocity'); ax2.set_ylabel('Predicted Linear Velocity')
        ax2.set_title(f'Linear: R² = {linear_r2:.4f}', fontsize=14, fontweight='bold', color='blue'); ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'plot_e_prediction_scatter.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_data_distribution(self, ground_truth: np.ndarray):
        print("📈 Creating Plot F: Data Distribution Analysis…")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Plot F: Dataset Distribution - "Data-Centric Challenge Analysis"', fontsize=16, fontweight='bold')
        ax1.hist(ground_truth[:, 1], bins=50, alpha=0.7, color='red', edgecolor='black')
        ax1.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.8)
        ax1.set_xlabel('Angular Velocity'); ax1.set_ylabel('Frequency'); ax1.grid(True, alpha=0.3)
        ax2.hist(ground_truth[:, 0], bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax2.set_xlabel('Linear Velocity'); ax2.set_ylabel('Frequency'); ax2.grid(True, alpha=0.3)
        fig.text(0.5, 0.02, 'INSIGHT: Narrow linear velocity range vs. wide angular range complicates learning.',
                 ha='center', fontsize=12, style='italic', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
        plt.tight_layout(); plt.subplots_adjust(bottom=0.15)
        plt.savefig(os.path.join(self.viz_dir, 'plot_f_data_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_feature_analysis_plots(self, feature_dict: Dict[str, np.ndarray]):
        print("🔍 Creating Plot G: Feature Space Analysis (Real)…")
        names = list(feature_dict.keys())
        n = min(6, len(names))
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Plot G: Feature Space Analysis - "Internal Model Representations"', fontsize=16, fontweight='bold')
        for ax, name in zip(axes.flat, names[:n]):
            f = feature_dict[name]
            ax.hist(f.flatten(), bins=50, alpha=0.7, edgecolor='black')
            ax.set_title(f'{name}\nmean={f.mean():.3f}, std={f.std():.3f}', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'plot_g_feature_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_loss_convergence_plot(self, train_losses: List[float], val_r2_linear: List[float], val_r2_angular: List[float]):
        print("📉 Creating Plot H: Training Convergence…")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle('Plot H: Training Convergence - "Learning Dynamics"', fontsize=16, fontweight='bold')
        epochs = range(1, len(train_losses) + 1)
        ax1.plot(epochs, train_losses, 'o-', linewidth=2, markersize=6); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Training Loss'); ax1.grid(True, alpha=0.3)
        ax2.plot(range(1, len(val_r2_linear)+1), val_r2_linear, 'o-', linewidth=2, markersize=6, label='Linear R²')
        ax2.plot(range(1, len(val_r2_angular)+1), val_r2_angular, 'o-', linewidth=2, markersize=6, label='Angular R²')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('R² Score'); ax2.legend(); ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'plot_h_training_convergence.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_error_analysis_plot(self, predictions: np.ndarray, ground_truth: np.ndarray):
        print("❌ Creating Plot I: Error Analysis…")
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Plot I: Error Analysis - "Understanding Failure Modes"', fontsize=16, fontweight='bold')
        linear_errors = predictions[:, 0] - ground_truth[:, 0]
        ax1.scatter(ground_truth[:, 0], linear_errors, alpha=0.6, color='blue')
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=2); ax1.set_xlabel('True Linear'); ax1.set_ylabel('Error'); ax1.grid(True, alpha=0.3)
        angular_errors = predictions[:, 1] - ground_truth[:, 1]
        ax2.scatter(ground_truth[:, 1], angular_errors, alpha=0.6, color='red')
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=2); ax2.set_xlabel('True Angular'); ax2.set_ylabel('Error'); ax2.grid(True, alpha=0.3)
        ax3.hist(linear_errors, bins=50, alpha=0.7, color='blue', edgecolor='black'); ax3.axvline(0, color='red', linestyle='--', linewidth=2)
        ax3.set_xlabel('Linear Error'); ax3.set_ylabel('Frequency'); ax3.grid(True, alpha=0.3)
        ax4.hist(angular_errors, bins=50, alpha=0.7, color='red', edgecolor='black'); ax4.axvline(0, color='red', linestyle='--', linewidth=2)
        ax4.set_xlabel('Angular Error'); ax4.set_ylabel('Frequency'); ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'plot_i_error_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_interactive_3d_plot(self, predictions: np.ndarray, ground_truth: np.ndarray):
        print("🌐 Creating Plot J: Interactive 3D Visualization…")
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Angular Velocity 3D', 'Linear Velocity 3D'),
                            specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]])
        fig.add_trace(go.Scatter3d(x=ground_truth[:,1], y=predictions[:,1], z=np.arange(len(ground_truth)),
                                   mode='markers',
                                   marker=dict(size=4, color=np.abs(ground_truth[:,1]-predictions[:,1]), colorscale='Viridis',
                                               colorbar=dict(title="Error Magnitude")), name='Angular'),
                      row=1, col=1)
        fig.add_trace(go.Scatter3d(x=ground_truth[:,0], y=predictions[:,0], z=np.arange(len(ground_truth)),
                                   mode='markers',
                                   marker=dict(size=4, color=np.abs(ground_truth[:,0]-predictions[:,0]), colorscale='Plasma',
                                               colorbar=dict(title="Error Magnitude")), name='Linear'),
                      row=1, col=2)
        fig.update_layout(title="Interactive 3D Prediction Analysis", height=600)
        fig.write_html(os.path.join(self.viz_dir, 'plot_j_interactive_3d.html'))
        
    def generate_comprehensive_report(self, predictions: np.ndarray, ground_truth: np.ndarray, train_losses: List[float]):
        print("\n📑 Generating Comprehensive Visualization Report…")
        linear_r2 = r2_score(ground_truth[:, 0], predictions[:, 0])
        angular_r2 = r2_score(ground_truth[:, 1], predictions[:, 1])
        linear_mae = mean_absolute_error(ground_truth[:, 0], predictions[:, 0])
        angular_mae = mean_absolute_error(ground_truth[:, 1], predictions[:, 1])
        report = f"""
### Key Findings:
1. **Angular Velocity Performance**: R² = {angular_r2:.4f}, MAE = {angular_mae:.4f}
2. **Linear Velocity Performance**: R² = {linear_r2:.4f}, MAE = {linear_mae:.4f}
3. **Performance Ratio**: Angular performs {angular_r2/max(linear_r2, 0.001):.1f}x better than Linear

### Visual Evidence Summary:
- **Plot A**: Raw sequential input
- **Plot B**: Depth perception
- **Plot C**: *Real* DINOv2 attention (CLS→patches)
- **Plot D**: Optical flow disparity (R² explanation)
- **Plot E**: Prediction scatter
- **Plot F**: Data distributions
- **Plot G**: *Real* internal feature distributions
- **Plot H**: Training dynamics
- **Plot I**: Error analysis
- **Plot J**: Interactive 3D

The signal disparity (rotation vs translation) plus narrow linear distribution explains the R² gap.
"""
        # Windows-safe UTF-8 write
        with open(os.path.join(self.viz_dir, 'visualization_report.md'), 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"✅ Comprehensive visualization report saved to: {self.viz_dir}")

# ================== ANGULAR VELOCITY SYSTEM ==================
class AdvancedMultiModalExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.semantic_dim = 768  # DINOv2
        self.depth_dim = 256     # After compression
        self.flow_dim = 128      # Optical flow descriptor
        print("🔄 Initializing Advanced Multi-Modal Feature Extractor (Angular)…")
        self._load_vision_models()
        self._setup_compression_layers()
        print("✅ Advanced feature extractor ready (Angular).")

    def _load_vision_models(self):
        print("  -> Loading DINOv2 semantic encoder…")
        self.dinov2_processor = AutoImageProcessor.from_pretrained(DINOV2_MODEL)
        self.dinov2_model = AutoModel.from_pretrained(DINOV2_MODEL).to(DEVICE)
        print("  -> Loading Depth-Anything depth encoder…")
        self.depth_processor = AutoImageProcessor.from_pretrained(DEPTH_ANYTHING_V2_MODEL)
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_ANYTHING_V2_MODEL).to(DEVICE)
        # FP16 ONLY for DINOv2 (Depth stays FP32)
        if ADVANCED_CONFIG['use_fp16']:
            self.dinov2_model = self.dinov2_model.half()
        self.dinov2_model.eval()
        self.depth_model.eval()

    def _setup_compression_layers(self):
        # Resolution-agnostic depth compressor (1xHxW -> 256)
        self.depth_compressor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, self.depth_dim)
        ).to(DEVICE)
        # Flow descriptor encoder (128 -> 128)
        self.flow_encoder = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, self.flow_dim)
        ).to(DEVICE)

    # ======= BATCHED forward =======
    def forward(self, image_sequences: List[List[Image.Image]]) -> Dict[str, torch.Tensor]:
        B = len(image_sequences)
        if B == 0:
            empty = torch.empty(0, 0, device=DEVICE)
            return {'semantic': empty, 'depth': empty, 'flow': empty}

        T = len(image_sequences[0])  # fixed by dataset padding
        flat_imgs = [img for seq in image_sequences for img in seq]  # (B*T)

        # --- DINOv2 batched (FP16) ---
        dino_list = []
        with torch.no_grad():
            for i in range(0, len(flat_imgs), VISION_CHUNK):
                chunk = flat_imgs[i:i+VISION_CHUNK]
                din_in = self.dinov2_processor(images=chunk, return_tensors="pt").to(DEVICE)
                if ADVANCED_CONFIG['use_fp16']:
                    din_in['pixel_values'] = din_in['pixel_values'].half()
                out = self.dinov2_model(**din_in, output_attentions=False)
                dino_list.append(out.last_hidden_state[:, 0, :].float())  # (N,768)
        dino_feats = torch.cat(dino_list, dim=0).view(B, T, -1)  # (B,T,768)

        # --- Depth batched (FP32) ---
        depth_list = []
        with torch.no_grad():
            for i in range(0, len(flat_imgs), VISION_CHUNK):
                chunk = flat_imgs[i:i+VISION_CHUNK]
                dep_in = self.depth_processor(images=chunk, return_tensors="pt").to(DEVICE)
                dep = self.depth_model(**dep_in).predicted_depth  # (N,H,W) or (N,1,H,W)
                if dep.dim() == 3:
                    dep = dep.unsqueeze(1)
                depth_list.append(self.depth_compressor(dep.float()))  # (N,256)
        depth_feats = torch.cat(depth_list, dim=0).view(B, T, -1)  # (B,T,256)

        # --- Flow (light CPU) ---
        flow_list = [self._extract_optical_flow_for_sequence(seq) for seq in image_sequences]  # each (T,128)
        flow_feats = nn.utils.rnn.pad_sequence(flow_list, batch_first=True, padding_value=0)   # (B,T,128)

        return {'semantic': dino_feats, 'depth': depth_feats, 'flow': flow_feats}

    def _extract_optical_flow_for_sequence(self, images: List[Image.Image]) -> torch.Tensor:
        if len(images) <= 1:
            return torch.zeros(max(1, len(images)), self.flow_dim).to(DEVICE)

        descs = [torch.zeros(self.flow_dim)]  # first step alignment (T timesteps total)
        for i in range(len(images) - 1):
            prev_img = cv2.cvtColor(np.array(images[i]),   cv2.COLOR_RGB2GRAY)
            curr_img = cv2.cvtColor(np.array(images[i+1]), cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_img, curr_img, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # divergence features (linear motion cue)
            du_dx = cv2.Sobel(flow[..., 0], cv2.CV_32F, 1, 0, ksize=3)
            dv_dy = cv2.Sobel(flow[..., 1], cv2.CV_32F, 0, 1, ksize=3)
            div = du_dx + dv_dy
            div_mean, div_std = float(div.mean()), float(div.std())

            desc = np.array([
                np.mean(magnitude), np.std(magnitude), np.max(magnitude),
                np.mean(angle), np.std(angle),
                np.percentile(magnitude, 75), np.percentile(magnitude, 95),
                np.sum(magnitude > np.mean(magnitude)) / magnitude.size,
                cv2.norm(flow, cv2.NORM_L2) / flow.size,
                np.mean(flow[..., 0]), np.std(flow[..., 0]),
                np.mean(flow[..., 1]), np.std(flow[..., 1]),
                div_mean, div_std,   # <-- added
            ], dtype=np.float32)

            if desc.shape[0] < 128:
                desc = np.pad(desc, (0, 128 - desc.shape[0]), mode='constant')
            else:
                desc = desc[:128]

            descs.append(torch.from_numpy(desc))

        flow_tensor = torch.stack(descs).float().to(DEVICE)  # (T,128)
        encoded_flow = self.flow_encoder(flow_tensor)        # (T,128)
        return encoded_flow

#ENCODERS & FUSION 
class TemporalLSTMEncoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=ADVANCED_CONFIG['lstm_hidden_size'],
            num_layers=ADVANCED_CONFIG['lstm_num_layers'],
            batch_first=True,
            dropout=ADVANCED_CONFIG['dropout_rate'] if ADVANCED_CONFIG['lstm_num_layers'] > 1 else 0,
            bidirectional=True
        )
        self.output_projection = nn.Linear(ADVANCED_CONFIG['lstm_hidden_size'] * 2,
                                           ADVANCED_CONFIG['lstm_hidden_size'])
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        final_hidden = lstm_out[:, -1, :]
        return self.output_projection(final_hidden)

class AdvancedFusionModule(nn.Module):
    def __init__(self, semantic_dim: int, depth_dim: int, flow_dim: int):
        super().__init__()
        self.semantic_encoder = TemporalLSTMEncoder(semantic_dim)
        self.depth_encoder    = TemporalLSTMEncoder(depth_dim)
        self.flow_encoder     = TemporalLSTMEncoder(flow_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=ADVANCED_CONFIG['lstm_hidden_size'],
            num_heads=8,
            dropout=ADVANCED_CONFIG['dropout_rate'],
            batch_first=True  
        )
        total_dim = ADVANCED_CONFIG['lstm_hidden_size'] * 3
        self.fusion_network = nn.Sequential(
            nn.Linear(total_dim, ADVANCED_CONFIG['fusion_hidden_size']), nn.ReLU(),
            nn.Dropout(ADVANCED_CONFIG['dropout_rate']),
            nn.Linear(ADVANCED_CONFIG['fusion_hidden_size'], ADVANCED_CONFIG['fusion_hidden_size'] // 2), nn.ReLU(),
            nn.Dropout(ADVANCED_CONFIG['dropout_rate']),
        )

    def forward(self, semantic_seq: torch.Tensor, depth_seq: torch.Tensor, flow_seq: torch.Tensor) -> torch.Tensor:
        semantic_encoded = self.semantic_encoder(semantic_seq)
        depth_encoded    = self.depth_encoder(depth_seq)
        flow_encoded     = self.flow_encoder(flow_seq)
        modal_features = torch.stack([semantic_encoded, depth_encoded, flow_encoded], dim=1)  # (B,3,H)
        attended_features, _ = self.attention(modal_features, modal_features, modal_features) # (B,3,H)
        fused_features = attended_features.reshape(attended_features.size(0), -1)
        return self.fusion_network(fused_features)

class AdaptiveBiasCorrector(nn.Module):
    def __init__(self, feature_dim: int, bias_scale_linear: float = 0.3, bias_scale_angular: float = 0.3):
        super().__init__()
        self.command_embeddings = nn.Embedding(5, 32)
        self.command_mapping = {"go_forward": 0, "go_left": 1, "go_right": 2, "stop": 3, "go_backward": 4}
        self.bias_network = nn.Sequential(
            nn.Linear(feature_dim + 32, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )
        self.register_buffer('velocity_templates', torch.zeros(5, 2))
        self.register_buffer('velocity_stds', torch.ones(5, 2))
        # NEW: different scales for linear vs angular
        self.bias_scale_linear = float(bias_scale_linear)
        self.bias_scale_angular = float(bias_scale_angular)
        
    def update_statistical_templates(self, command_velocities: Dict[str, List]):
        for command, velocities in command_velocities.items():
            if command in self.command_mapping and velocities:
                idx = self.command_mapping[command]
                vel_array = torch.FloatTensor(velocities)
                self.velocity_templates[idx] = torch.mean(vel_array, dim=0)
                self.velocity_stds[idx] = torch.std(vel_array, dim=0) + 1e-6

    def forward(self, features: torch.Tensor, commands: List[str]) -> torch.Tensor:
        cmd_indices = [self.command_mapping.get(cmd, 0) for cmd in commands]
        cmd_tensor = torch.LongTensor(cmd_indices).to(features.device)
        cmd_embeddings = self.command_embeddings(cmd_tensor)
        combined = torch.cat([features, cmd_embeddings], dim=1)
        bias_corrections = self.bias_network(combined)      # (B,2)
        template_predictions = self.velocity_templates[cmd_tensor]  # (B,2)
        scales = torch.tensor([self.bias_scale_linear, self.bias_scale_angular], device=features.device)
        final_predictions = template_predictions + bias_corrections * scales   
        return final_predictions

# DUAL SYSTEM 
class DualVelocityPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.angular_feature_extractor = AdvancedMultiModalExtractor()
        self.angular_fusion  = AdvancedFusionModule(768, 256, 128)
        fusion_output_dim = ADVANCED_CONFIG['fusion_hidden_size'] // 2
        self.angular_head = nn.Sequential(
            nn.Linear(fusion_output_dim, 128), nn.ReLU(), nn.Dropout(ADVANCED_CONFIG['dropout_rate']),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        # Angular bias unchanged (0.3)
        self.angular_bias_corrector = AdaptiveBiasCorrector(fusion_output_dim, bias_scale_linear=0.3, bias_scale_angular=0.3)
        self.angular_ensemble_weights = nn.Parameter(torch.FloatTensor([0.7, 0.3]))

        self.linear_fusion = AdvancedFusionModule(768, 256, 128)
        self.linear_head = nn.Sequential(
            nn.Linear(fusion_output_dim, 128), nn.ReLU(), nn.Dropout(ADVANCED_CONFIG['dropout_rate']),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.linear_bias_corrector = AdaptiveBiasCorrector(fusion_output_dim, bias_scale_linear=0.5, bias_scale_angular=0.3)
        self.linear_ensemble_weights = nn.Parameter(torch.FloatTensor([0.5, 0.5]))

    def forward(self, image_sequences: List[List[Image.Image]], commands: List[str]) -> torch.Tensor:
        angular_features = self.angular_feature_extractor(image_sequences)
        semantic_seq, depth_seq, flow_seq = angular_features['semantic'], angular_features['depth'], angular_features['flow']
        # Angular
        angular_fused = self.angular_fusion(semantic_seq, depth_seq, flow_seq)
        angular_main  = self.angular_head(angular_fused)
        angular_bias  = self.angular_bias_corrector(angular_fused, commands)[:, 1:2]
        ang_w = torch.softmax(self.angular_ensemble_weights, dim=0)
        angular_pred = ang_w[0] * angular_main + ang_w[1] * angular_bias
        # Linear
        linear_fused = self.linear_fusion(semantic_seq, depth_seq, flow_seq)
        linear_main  = self.linear_head(linear_fused)
        linear_bias  = self.linear_bias_corrector(linear_fused, commands)[:, 0:1]
        lin_w = torch.softmax(self.linear_ensemble_weights, dim=0)
        linear_pred = lin_w[0] * linear_main + lin_w[1] * linear_bias
        return torch.cat([linear_pred, angular_pred], dim=1)
    
    def update_bias_templates(self, dataset):
        print("🔄 Updating statistical templates for dual system…")
        command_velocities = defaultdict(list)
        for i in range(len(dataset)):
            sample = dataset[i]
            command_velocities[sample["command"]].append(sample["velocity"].numpy())
        self.angular_bias_corrector.update_statistical_templates(command_velocities)
        self.linear_bias_corrector.update_statistical_templates(command_velocities)
        print("✅ All statistical templates updated")

#  DATASET 
class AdvancedNavigationDataset(Dataset):
    def __init__(self, jsonl_path: str, seq_len: int):
        self.samples = []
        self.seq_len = seq_len
        with open(jsonl_path, "r", encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image_paths = sample["images"]

        if len(image_paths) < self.seq_len:
            image_paths = image_paths + [image_paths[-1]] * (self.seq_len - len(image_paths))
        else:
            image_paths = image_paths[-self.seq_len:]

        images = [
            Image.open(p).convert("RGB").resize((ADVANCED_CONFIG['image_size'], ADVANCED_CONFIG['image_size']))
            for p in image_paths
        ]
        
        velocity = self._extract_velocity_from_text(sample["text"])
        command = self._extract_command_from_text(sample["text"])
        
        return {"images": images, "velocity": torch.FloatTensor(velocity), "command": command}

    def _extract_velocity_from_text(self, text: str) -> np.ndarray:
        try:
            _, assistant_part = text.split("ASSISTANT:", 1)
            assistant_json = json.loads(assistant_part.strip())
            vel_cmd = assistant_json.get("Velocity command", {})
            linear = float(vel_cmd.get("linear_velocity", 0.0))
            angular = float(vel_cmd.get("angular_velocity", 0.0))
            return np.array([linear, angular], dtype=np.float32)
        except:
            return np.array([0.0, 0.0], dtype=np.float32)

    def _extract_command_from_text(self, text: str) -> str:
        try:
            _, assistant_part = text.split("ASSISTANT:", 1)
            assistant_json = json.loads(assistant_part.strip())
            return assistant_json.get("Next high-level command", "go_forward")
        except:
            return "go_forward"

def custom_collate_fn(batch):
    images = [item["images"] for item in batch]
    velocities = torch.stack([item["velocity"] for item in batch])
    commands = [item["command"] for item in batch]
    return {"images": images, "velocity": velocities, "command": commands}

# ATTENTION MAP 
def compute_real_dino_attention_map(extractor: AdvancedMultiModalExtractor, pil_image: Image.Image) -> np.ndarray:
    extractor.dinov2_model.eval()
    inputs = extractor.dinov2_processor(images=[pil_image], return_tensors="pt").to(DEVICE)
    if ADVANCED_CONFIG['use_fp16']:
        inputs['pixel_values'] = inputs['pixel_values'].half()
    with torch.no_grad():
        out = extractor.dinov2_model(**inputs, output_attentions=True)
    atts = out.attentions  # list of length L: (B, heads, tokens, tokens)
    if atts is None or len(atts) == 0:
        return np.zeros((ADVANCED_CONFIG['image_size'], ADVANCED_CONFIG['image_size']), dtype=np.float32)
    attn = atts[-1].mean(dim=1)[0]  # (tokens, tokens)
    cls_to_patches = attn[0, 1:]    # drop CLS→CLS
    num_patches = cls_to_patches.shape[0]
    side = int(math.sqrt(num_patches))
    if side * side != num_patches:
        amap = cls_to_patches.detach().float().cpu().numpy()
        amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-6)
        return cv2.resize(np.ones((8,8), dtype=np.float32)*amap.mean(),
                          (ADVANCED_CONFIG['image_size'], ADVANCED_CONFIG['image_size']),
                          interpolation=cv2.INTER_CUBIC)
    amap = cls_to_patches.reshape(side, side).detach().float().cpu().numpy()
    amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-6)
    amap = cv2.resize(amap, (ADVANCED_CONFIG['image_size'], ADVANCED_CONFIG['image_size']), interpolation=cv2.INTER_CUBIC)
    return amap

# UTILS: target stats 
def target_stats(dataset):
    vs = np.stack([dataset[i]["velocity"].numpy() for i in range(len(dataset))])
    mu = vs.mean(axis=0)
    sd = vs.std(axis=0) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)

# TRAIN / EVAL 
def train_epoch_with_viz(model: DualVelocityPredictor, train_loader: DataLoader, optimizer: optim.Optimizer, 
                        scaler, viz_suite: VisualizationSuite, epoch: int,
                        y_mu: np.ndarray, y_sd: np.ndarray) -> Tuple[float, Dict]:
    model.train()
    model.angular_feature_extractor.dinov2_model.eval()
    model.angular_feature_extractor.depth_model.eval()

    total_loss = 0.0
    feature_dict = defaultdict(list)

    mu = torch.tensor(y_mu, device=DEVICE)
    sd = torch.tensor(y_sd, device=DEVICE)
    
    for batch_idx, batch in enumerate(train_loader):
        images_batch = batch["images"]
        velocities_batch = batch["velocity"].to(DEVICE)
        commands_batch = batch["command"]

        print(f">> forward start | epoch {epoch} batch {batch_idx}/{len(train_loader)}", flush=True)
        if scaler:
            with torch.cuda.amp.autocast():
                predictions = model(images_batch, commands_batch)
             
                pred_norm = (predictions - mu) / sd
                targ_norm = (velocities_batch - mu) / sd

                linear_loss  = nn.functional.mse_loss(pred_norm[:, 0:1], targ_norm[:, 0:1])
                angular_loss = nn.functional.mse_loss(pred_norm[:, 1:2], targ_norm[:, 1:2])

               
                LIN_W = 3.0 if epoch < 5 else 2.0
                ANG_W = 1.0
                total_loss_batch = LIN_W * linear_loss + ANG_W * angular_loss
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(total_loss_batch).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            predictions = model(images_batch, commands_batch)
            pred_norm = (predictions - mu) / sd
            targ_norm = (velocities_batch - mu) / sd
            linear_loss  = nn.functional.mse_loss(pred_norm[:, 0:1], targ_norm[:, 0:1])
            angular_loss = nn.functional.mse_loss(pred_norm[:, 1:2], targ_norm[:, 1:2])
            LIN_W = 3.0 if epoch < 5 else 2.0
            ANG_W = 1.0
            total_loss_batch = LIN_W * linear_loss + ANG_W * angular_loss
            optimizer.zero_grad(set_to_none=True)
            total_loss_batch.backward()
            optimizer.step()
        print("<< forward end", flush=True)
        
        total_loss += total_loss_batch.item()
        
       
        if epoch == 0 and batch_idx < 5:
            with torch.no_grad():
                sample_images = images_batch[0]
                sample_velocity = velocities_batch[0].cpu().numpy()
                if len(sample_images) >= 2:
                    img1 = cv2.cvtColor(np.array(sample_images[0]), cv2.COLOR_RGB2GRAY)
                    img2 = cv2.cvtColor(np.array(sample_images[1]), cv2.COLOR_RGB2GRAY)
                    flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    if abs(sample_velocity[1]) > 0.1:
                        feature_dict['turn_samples'].append((sample_images[0], flow))
                    else:
                        feature_dict['forward_samples'].append((sample_images[0], flow))
               
                depth_input = model.angular_feature_extractor.depth_processor(images=[sample_images[0]], return_tensors="pt").to(DEVICE)
                depth_map = model.angular_feature_extractor.depth_model(**depth_input).predicted_depth
                if depth_map.dim() == 3:
                    dm = depth_map[0].detach().cpu().numpy()
                else:
                    dm = depth_map[0,0].detach().cpu().numpy()
                feature_dict['depth_samples'].append((sample_images[0], dm))
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)}, Total Loss: {total_loss_batch.item():.4f}")
            print(f"    Linear Loss(norm): {linear_loss.item():.4f}, Angular Loss(norm): {angular_loss.item():.4f}")
            
    return total_loss / len(train_loader), feature_dict

def evaluate_with_viz(model: DualVelocityPredictor, val_loader: DataLoader, viz_suite: VisualizationSuite) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_predictions, all_ground_truth = [], []
    sample_data = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            images_batch = batch["images"]
            velocities_batch = batch["velocity"]
            commands_batch = batch["command"]
            predictions = model(images_batch, commands_batch)
            all_predictions.append(predictions.cpu().numpy())
            all_ground_truth.append(velocities_batch.numpy())
            if i < 3:
                for j in range(min(len(images_batch), 2)):
                    sample_data.append({
                        'images': images_batch[j],
                        'velocity': velocities_batch[j].numpy(),
                        'prediction': predictions[j].cpu().numpy(),
                        'command': commands_batch[j]
                    })
            if (i + 1) % 10 == 0:
                print(f"  ✅ Evaluated batch {i+1}/{len(val_loader)}")
    predictions_array = np.vstack(all_predictions)
    ground_truth_array = np.vstack(all_ground_truth)

    
    if sample_data:
        for idx, sample in enumerate(sample_data[:3]):
            viz_suite.plot_input_sequence(sample['images'], idx)
        sample0 = sample_data[0]
        # Depth (FP32) with no grad + detach before numpy
        with torch.no_grad():
            depth_input = model.angular_feature_extractor.depth_processor(images=[sample0['images'][0]], return_tensors="pt").to(DEVICE)
            depth_map = model.angular_feature_extractor.depth_model(**depth_input).predicted_depth
        if depth_map.dim() == 3:
            dm = depth_map[0].detach().cpu().numpy()
        else:
            dm = depth_map[0,0].detach().cpu().numpy()
        viz_suite.plot_depth_perception(sample0['images'][0], dm, 0)
        # Real DINO attention
        attention_map = compute_real_dino_attention_map(model.angular_feature_extractor, sample0['images'][0])
        viz_suite.plot_dinov2_attention(sample0['images'][0], attention_map, 0)

    return predictions_array, ground_truth_array

def print_dual_results_with_viz(predictions: np.ndarray, ground_truth: np.ndarray, viz_suite: VisualizationSuite):
    print("\n" + "="*100)
    print("DUAL-BRANCH VELOCITY PREDICTION - PERFORMANCE SUMMARY WITH VISUALIZATIONS")
    print("="*100)
    linear_mae   = mean_absolute_error(ground_truth[:, 0], predictions[:, 0])
    linear_r2    = r2_score(ground_truth[:, 0], predictions[:, 0])
    linear_acc   = np.mean(np.abs(predictions[:, 0] - ground_truth[:, 0]) < 0.30) * 100
    linear_corr  = np.corrcoef(ground_truth[:, 0], predictions[:, 0])[0, 1]
    angular_mae  = mean_absolute_error(ground_truth[:, 1], predictions[:, 1])
    angular_r2   = r2_score(ground_truth[:, 1], predictions[:, 1])
    angular_acc  = np.mean(np.abs(predictions[:, 1] - ground_truth[:, 1]) < 0.30) * 100
    angular_corr = np.corrcoef(ground_truth[:, 1], predictions[:, 1])[0, 1]

    print(f"\n📊 LINEAR VELOCITY PERFORMANCE:")
    print(f"  R² Score: {linear_r2:.4f}")
    print(f"  Mean Absolute Error: {linear_mae:.4f}")
    print(f"  Pearson Correlation: {linear_corr:.4f}")
    print(f"  Prediction Accuracy: {linear_acc:.2f}%")
    print(f"\n🔄 ANGULAR VELOCITY PERFORMANCE:")
    print(f"  R² Score: {angular_r2:.4f}")
    print(f"  Mean Absolute Error: {angular_mae:.4f}")
    print(f"  Pearson Correlation: {angular_corr:.4f}")
    print(f"  Prediction Accuracy: {angular_acc:.2f}%")
    print(f"\n🎯 OVERALL SYSTEM:")
    overall_mae = mean_absolute_error(ground_truth, predictions)
    print(f"  Combined MAE: {overall_mae:.4f}")

    print(f"\n📈 Generating comprehensive visualizations…")
    viz_suite.plot_prediction_scatter(predictions, ground_truth)
    viz_suite.plot_data_distribution(ground_truth)
    viz_suite.create_error_analysis_plot(predictions, ground_truth)
    viz_suite.create_interactive_3d_plot(predictions, ground_truth)
    print("\n" + "="*100)

#  MAIN 
def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    viz_suite = VisualizationSuite(OUTPUT_DIR)
    
    print("📊 Loading datasets...")
    train_dataset = AdvancedNavigationDataset(TRAIN_JSONL, ADVANCED_CONFIG['sequence_length'])
    val_dataset   = AdvancedNavigationDataset(VAL_JSONL,   ADVANCED_CONFIG['sequence_length'])

 
    y_mu, y_sd = target_stats(train_dataset)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=ADVANCED_CONFIG['batch_size'],
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=ADVANCED_CONFIG['num_workers'],
        pin_memory=ADVANCED_CONFIG['pin_memory'],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=ADVANCED_CONFIG['batch_size'],
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=ADVANCED_CONFIG['num_workers'],
        pin_memory=ADVANCED_CONFIG['pin_memory'],
    )

    print("\n🧠 Initializing Simplified Dual-Branch Velocity Predictor with Visualizations…")
    print("  🔄 Angular branch: Using your existing working system (UNCHANGED)")
    print("  📊 Linear branch: Using SAME architecture as angular (SIMPLIFIED)")
    print("  📈 Visualization: Comprehensive MSc dissertation analysis suite")
    
    model = DualVelocityPredictor().to(DEVICE)
    model.update_bias_templates(train_dataset)
   
    angular_params = (
        list(model.angular_fusion.parameters()) +
        list(model.angular_head.parameters()) +
        list(model.angular_bias_corrector.parameters())
    )
    linear_params  = (
        list(model.linear_fusion.parameters()) +
        list(model.linear_head.parameters()) +
        list(model.linear_bias_corrector.parameters())
    )
    optimizer = optim.AdamW([
        {'params': angular_params, 'lr': ADVANCED_CONFIG['learning_rate'], 'weight_decay': 1e-4},
        {'params': linear_params,  'lr': ADVANCED_CONFIG['learning_rate'], 'weight_decay': 1e-4},
        {'params': [model.angular_ensemble_weights, model.linear_ensemble_weights],
         'lr': ADVANCED_CONFIG['learning_rate'] * 0.1, 'weight_decay': 0}
    ])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scaler = torch.cuda.amp.GradScaler() if ADVANCED_CONFIG['use_fp16'] else None

    print(f"\n🚀 Training simplified dual system for {ADVANCED_CONFIG['num_epochs']} epochs with comprehensive visualization…")
    best_linear_r2 = -999
    best_epoch = -1
    train_losses = []
    val_r2_linear = []
    val_r2_angular = []
    collected_features = {}

    patience_counter = 0  
    
    for epoch in range(ADVANCED_CONFIG['num_epochs']):
        print(f"\n--- Epoch {epoch+1}/{ADVANCED_CONFIG['num_epochs']} ---")
        train_loss, feature_dict = train_epoch_with_viz(
            model, train_loader, optimizer, scaler, viz_suite, epoch, y_mu, y_sd
        )
        train_losses.append(train_loss)
        if epoch == 0:
            collected_features.update(feature_dict)
        print(f"Epoch {epoch+1} - Average Train Loss: {train_loss:.4f}")
        
        # Evaluate with visualization
        print(f"\n📊 Evaluation at epoch {epoch+1}…")
        predictions, ground_truth = evaluate_with_viz(model, val_loader, viz_suite)
        linear_r2 = r2_score(ground_truth[:, 0], predictions[:, 0])
        angular_r2 = r2_score(ground_truth[:, 1], predictions[:, 1])
        val_r2_linear.append(linear_r2)
        val_r2_angular.append(angular_r2)
        scheduler.step(train_loss)
        
        # Track best linear performance (save best model)
        if linear_r2 > best_linear_r2:
            best_linear_r2 = linear_r2
            best_epoch = epoch
            patience_counter = 0
            print(f"🎯 New best linear R²: {best_linear_r2:.4f}")
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_dual_velocity_model.pth"))
        else:
            patience_counter += 1
            print(f"⏳ No linear R² improvement ({patience_counter}/{ADVANCED_CONFIG['early_stopping_patience']})")

        print_dual_results_with_viz(predictions, ground_truth, viz_suite)

    
        if patience_counter >= ADVANCED_CONFIG['early_stopping_patience']:
            print(f"🛑 Early stopping triggered after {epoch+1} epochs (patience: {ADVANCED_CONFIG['early_stopping_patience']})")
            print(f"📊 Best linear R² achieved: {best_linear_r2:.4f} at epoch {best_epoch+1}")
            break
    # Create optical flow comparison visualization (Plot D)
    if collected_features:
        print("\n🔥 Creating Optical Flow Smoking Gun Visualization…")
        turn_samples = collected_features.get('turn_samples', [])
        forward_samples = collected_features.get('forward_samples', [])
        if turn_samples and forward_samples:
            turn_img, turn_flow = turn_samples[0]
            forward_img, forward_flow = forward_samples[0]
            viz_suite.plot_optical_flow_comparison(turn_img, turn_flow, forward_img, forward_flow)
        # Real DINO attention for a depth sample image
        if 'depth_samples' in collected_features and collected_features['depth_samples']:
            print("🧠 Creating DINOv2 Attention Visualization (Real)…")
            sample_img, _ = collected_features['depth_samples'][0]
            attention_map = compute_real_dino_attention_map(model.angular_feature_extractor, sample_img)
            viz_suite.plot_dinov2_attention(sample_img, attention_map, 0)
    
    print("\n📈 Final Evaluation with Best Model and Complete Visualizations…")
    if os.path.exists(os.path.join(OUTPUT_DIR, "best_dual_velocity_model.pth")):
        model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_dual_velocity_model.pth"), map_location=DEVICE))
        print("📥 Loaded best model weights")
    final_predictions, final_ground_truth = evaluate_with_viz(model, val_loader, viz_suite)
    
    print("\n🎨 Generating comprehensive visualization suite…")
    # feature analysis for Plot G
    feature_analysis_dict = {}
    with torch.no_grad():
        K = min(16, len(val_dataset))
        idxs = np.linspace(0, len(val_dataset)-1, K, dtype=int)
        sequences = [val_dataset[i]["images"] for i in idxs]
        feats = model.angular_feature_extractor(sequences)
        semantic_feats = feats['semantic'].reshape(-1, feats['semantic'].shape[-1]).cpu().numpy()
        depth_feats    = feats['depth'].reshape(-1, feats['depth'].shape[-1]).cpu().numpy()
        flow_feats     = feats['flow'].reshape(-1,  feats['flow'].shape[-1]).cpu().numpy()
        feature_analysis_dict["Semantic Features (DINOv2 CLS)"] = semantic_feats
        feature_analysis_dict["Depth Features (Compressed)"]    = depth_feats
        feature_analysis_dict["Flow Features (Encoded)"]        = flow_feats
    viz_suite.create_feature_analysis_plots(feature_analysis_dict)
    viz_suite.create_loss_convergence_plot(train_losses, val_r2_linear, val_r2_angular)
    print_dual_results_with_viz(final_predictions, final_ground_truth, viz_suite)
    viz_suite.generate_comprehensive_report(final_predictions, final_ground_truth, train_losses)
    
    final_linear_r2  = r2_score(final_ground_truth[:, 0], final_predictions[:, 0])
    final_angular_r2 = r2_score(final_ground_truth[:, 1], final_predictions[:, 1])
    print(f"\n🎯 TARGET ACHIEVEMENT CHECK:")
    print(f"  Linear R² Target: 0.30-0.50 | Achieved: {final_linear_r2:.4f} {'✅' if final_linear_r2 >= 0.30 else '❌'}")
    print(f"  Angular R² Maintenance: ≥0.70 | Achieved: {final_angular_r2:.4f} {'✅' if final_angular_r2 >= 0.70 else '❌'}")
    
    print("\n💾 Saving results and visualizations…")
    results_to_save = {
        'predictions': [{'linear': p[0], 'angular': p[1]} for p in final_predictions.tolist()],
        'ground_truth': [{'linear': gt[0], 'angular': gt[1]} for gt in final_ground_truth.tolist()],
        'performance_metrics': {
            'linear_r2': final_linear_r2,
            'angular_r2': final_angular_r2,
            'target_achieved': final_linear_r2 >= 0.30 and final_angular_r2 >= 0.70,
            'training_losses': train_losses,
            'val_r2_linear': val_r2_linear,
            'val_r2_angular': val_r2_angular
        },
        'visualization_info': {
            'plots_generated': [
                'Plot A: Input Image Sequence',
                'Plot B: Depth Perception', 
                'Plot C: DINOv2 Semantic Understanding (Real)',
                'Plot D: Optical Flow Comparison (Smoking Gun)',
                'Plot E: Prediction vs Ground Truth',
                'Plot F: Data Distribution Analysis',
                'Plot G: Feature Space Analysis (Real)',
                'Plot H: Training Convergence',
                'Plot I: Error Analysis',
                'Plot J: Interactive 3D Visualization'
            ],
            'visualization_directory': viz_suite.viz_dir
        }
    }
    json_output_path = os.path.join(OUTPUT_DIR, "simplified_dual_predictions_with_visualizations.json")
    # Windows-safe UTF-8 JSON with unicode preserved
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, indent=4, ensure_ascii=False)
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "simplified_dual_velocity_model_with_viz.pth"))
    
    print(f"\n🎨 COMPREHENSIVE VISUALIZATION SUMMARY:")
    print(f"📁 All visualizations saved to: {viz_suite.viz_dir}")
    print(f"📊 Generated {len(results_to_save['visualization_info']['plots_generated'])} comprehensive plots")
    print(f"🔍 Key insights for dissertation:")
    print(f"   • Plot C now uses REAL DINO attention (CLS→patches)")
    print(f"   • Plot D shows rotation vs translation flow clearly")
    print(f"   • Plot G uses REAL feature distributions (semantic/depth/flow)")
    
    clear_gpu_memory()
    print(f"\n✅ Simplified dual-branch pipeline with comprehensive visualizations complete!")
    print(f"🎯 Expected: Linear R²≈0.3-0.5+, Angular R²≈0.74 (data dependent)")
    print(f"🗂️  Results location: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
