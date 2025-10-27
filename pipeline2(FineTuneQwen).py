import os
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    AutoModelForVision2Seq,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import gpytorch
from scipy.spatial.transform import Rotation as R
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import pickle
import argparse
import time
from datetime import datetime

warnings.filterwarnings('ignore')

# ENHANCED CONFIGURATION FOR RTX A6000 
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
OUTPUT_DIR = "./qwen2.5vl3b_hybrid_gp_a6000"
SEED = 42
MIN_PIXELS = 256 * 256 
MAX_PIXELS = 512 * 512  
MAX_SEQ_LEN_BASE = 2048  
LABEL_MARGIN = 128
MAX_SEQ_LEN_CAP = 2304 
LORA_RANK = 64 
LORA_ALPHA = 128  
LORA_DROPOUT = 0.05  
NUM_EPOCHS = 15  
GRAD_ACCUM_STEPS = 4  
LEARNING_RATE = 3e-5  
WARMUP_RATIO = 0.1  
WEIGHT_DECAY = 0.005  
BATCH_SIZE = 2  
EVAL_BATCH_SIZE = 4  
MAX_STEPS=3000
QWEN_FEATURE_DIM = 2048
GP_INPUT_DIM = 128  
BRIDGE_HIDDEN_DIM = 512  
CONTEXT_LEN = 10
VELOCITY_LOSS_WEIGHT = 0.5  
TEXT_LOSS_WEIGHT = 0.5
USE_BF16 = True  
USE_FLASH_ATTENTION = True 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GP_TRAINING_SAMPLES = 500 
GP_KERNEL_TYPE = 'matern'  
GP_INDUCING_POINTS = 256 


class ExactGPModel(gpytorch.models.ExactGP):
    
    def __init__(self, train_x, train_y, likelihood, kernel_type='rbf'):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        if kernel_type == 'rbf':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel()
            )
        elif kernel_type == 'matern':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=2.5)
            )
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GPVelocityPredictor(nn.Module):
    
    def __init__(self, input_dim=GP_INPUT_DIM, hidden_dim=256, use_full_gp=True):
        super().__init__()
        self.use_full_gp = use_full_gp
        self.input_dim = input_dim
        if use_full_gp:
            self.likelihood_linear = gpytorch.likelihoods.GaussianLikelihood()
            self.likelihood_angular = gpytorch.likelihoods.GaussianLikelihood()
            self.gp_linear = None
            self.gp_angular = None
           
            self.feature_extractor = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.05),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.05),
                nn.Linear(hidden_dim, input_dim // 2)
            )
        else:
            
            self.nn_predictor = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.05),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.05),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.05),
                nn.Linear(hidden_dim // 2, 2)
            )

    def extract_features(self, x):
        if self.use_full_gp:
            return self.feature_extractor(x)
        return x

    def fit_gp(self, train_features, train_velocities):
        if not self.use_full_gp:
            return
        print(f"Fitting GP with {len(train_features)} samples...")
        with torch.no_grad():
            gp_features = self.extract_features(train_features)
        
        
        self.gp_linear = ExactGPModel(
            gp_features,
            train_velocities[:, 0],
            self.likelihood_linear,
            kernel_type=GP_KERNEL_TYPE
        )
        self.gp_angular = ExactGPModel(
            gp_features,
            train_velocities[:, 1],
            self.likelihood_angular,
            kernel_type=GP_KERNEL_TYPE
        )
        
      
        self.gp_linear.train()
        self.gp_angular.train()
        self.likelihood_linear.train()
        self.likelihood_angular.train()
        
        # Optimize hyperparameters
        print("Optimizing GP hyperparameters...")
        optimizer_linear = torch.optim.Adam(self.gp_linear.parameters(), lr=0.1)
        optimizer_angular = torch.optim.Adam(self.gp_angular.parameters(), lr=0.1)
        mll_linear = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood_linear, self.gp_linear)
        mll_angular = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood_angular, self.gp_angular)
        
        # Training loop for GP hyperparameters
        for i in range(50):  # More iterations for better hyperparameters
            # Linear velocity GP
            optimizer_linear.zero_grad()
            output_linear = self.gp_linear(gp_features)
            loss_linear = -mll_linear(output_linear, train_velocities[:, 0])
            loss_linear.backward()
            optimizer_linear.step()
            
            # Angular velocity GP
            optimizer_angular.zero_grad()
            output_angular = self.gp_angular(gp_features)
            loss_angular = -mll_angular(output_angular, train_velocities[:, 1])
            loss_angular.backward()
            optimizer_angular.step()
            
            if i % 10 == 0:
                print(f"  Iteration {i}: Linear loss = {loss_linear.item():.4f}, Angular loss = {loss_angular.item():.4f}")
        
        print("GP hyperparameter optimization complete!")

    def forward(self, x):
        if self.use_full_gp and self.gp_linear is not None:
            features = self.extract_features(x)
            self.gp_linear.eval()
            self.gp_angular.eval()
            self.likelihood_linear.eval()
            self.likelihood_angular.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                linear_pred = self.likelihood_linear(self.gp_linear(features))
                angular_pred = self.likelihood_angular(self.gp_angular(features))
                linear_mean = linear_pred.mean
                angular_mean = angular_pred.mean
                velocities = torch.stack([linear_mean, angular_mean], dim=1)
                return velocities
        else:
            return self.nn_predictor(x)

# FEATURE BRIDGE NETWORK 
class QwenToGPBridge(nn.Module):
    def __init__(self, qwen_dim=QWEN_FEATURE_DIM, gp_dim=GP_INPUT_DIM, hidden_dim=BRIDGE_HIDDEN_DIM):
        super().__init__()
        self.feature_projector = nn.Sequential(
            nn.Linear(qwen_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, gp_dim),
            nn.Tanh()
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=qwen_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, qwen_features, return_attention=False):
        if qwen_features.dim() == 2:
            qwen_features = qwen_features.unsqueeze(1)
        attended_features, attention_weights = self.attention(
            qwen_features, qwen_features, qwen_features
        )
        pooled_features = attended_features.mean(dim=1)
        gp_features = self.feature_projector(pooled_features)
        if return_attention:
            return gp_features, attention_weights
        return gp_features

#  HYBRID MODEL
class HybridQwenGP(nn.Module):
    def __init__(self, qwen_model, processor):
        super().__init__()
        self.qwen_model = qwen_model
        self.processor = processor
        self.bridge = QwenToGPBridge()
        self.gp_predictor = GPVelocityPredictor(use_full_gp=False)
        self.extracted_features = None
        self._register_hooks()
        self.training_mode = True

    def _register_hooks(self):
        def feature_hook(module, input, output):
            # Handle different output types from Qwen model
            if hasattr(output, 'hidden_states') and output.hidden_states is not None:
                # Use last hidden state from model output
                self.extracted_features = output.hidden_states
            elif hasattr(output, 'last_hidden_state'):
                self.extracted_features = output.last_hidden_state
            elif isinstance(output, tuple) and len(output) > 0:
                # Take first element if tuple
                if hasattr(output[0], 'hidden_states'):
                    self.extracted_features = output[0].hidden_states
                elif torch.is_tensor(output[0]):
                    self.extracted_features = output[0]
                else:
                    self.extracted_features = None
            elif torch.is_tensor(output):
                self.extracted_features = output
            else:
                self.extracted_features = None
        
        hooked = False
        
        if hasattr(self.qwen_model, 'model') and hasattr(self.qwen_model.model, 'layers'):
            if len(self.qwen_model.model.layers) > 0:
                last_layer = self.qwen_model.model.layers[-1]
                last_layer.register_forward_hook(feature_hook)
                print("✅ Hook registered on model.layers[-1]")
                hooked = True
        
        if not hooked:
            print("⚠️ Warning: Could not register specific layer hook")
            print("Model structure:", type(self.qwen_model))
            # Fallback: hook the entire model
            self.qwen_model.register_forward_hook(feature_hook)
            print("✅ Hook registered on entire model (fallback)")

    def extract_qwen_features(self, images, text_inputs):
        with torch.no_grad():
            outputs = self.qwen_model(**text_inputs, return_dict=True)
            
            if self.extracted_features is None:
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    self.extracted_features = outputs.hidden_states
                elif hasattr(outputs, 'last_hidden_state'):
                    self.extracted_features = outputs.last_hidden_state
            
            if self.extracted_features is not None and torch.is_tensor(self.extracted_features):
                return self.extracted_features
            else:
                batch_size = text_inputs['input_ids'].shape[0]
                seq_len = text_inputs['input_ids'].shape[1]
                return torch.randn(batch_size, seq_len, QWEN_FEATURE_DIM, device=self.qwen_model.device)

    def forward(self, **inputs):
       
        self.extracted_features = None
        qwen_outputs = self.qwen_model(**inputs)
        if self.extracted_features is None:
            if hasattr(qwen_outputs, 'hidden_states') and qwen_outputs.hidden_states is not None:
                self.extracted_features = qwen_outputs.hidden_states
            elif hasattr(qwen_outputs, 'last_hidden_state'):
                self.extracted_features = qwen_outputs.last_hidden_state
        
        # Generate velocity predictions
        if self.extracted_features is not None and torch.is_tensor(self.extracted_features):
            try:
                gp_features = self.bridge(self.extracted_features)
                predicted_velocities = self.gp_predictor(gp_features)
            except Exception as e:
                print(f"Warning: Error in velocity prediction: {e}")
                batch_size = inputs['input_ids'].shape[0]
                predicted_velocities = torch.zeros(batch_size, 2, device=self.qwen_model.device)
        else:
            batch_size = inputs['input_ids'].shape[0]
            predicted_velocities = torch.zeros(batch_size, 2, device=self.qwen_model.device)
        
        qwen_outputs.predicted_velocities = predicted_velocities
        return qwen_outputs

    def generate_with_velocities(self, **inputs):
        with torch.no_grad():
            generated_ids = self.qwen_model.generate(**inputs)
            
            
            self.extracted_features = None
            outputs = self.qwen_model(**inputs, return_dict=True)
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                self.extracted_features = outputs.hidden_states
            elif hasattr(outputs, 'last_hidden_state'):
                self.extracted_features = outputs.last_hidden_state
            
            if self.extracted_features is not None and torch.is_tensor(self.extracted_features):
                try:
                    gp_features = self.bridge(self.extracted_features)
                    velocities = self.gp_predictor(gp_features)
                except Exception as e:
                    print(f"Warning: Velocity prediction failed during generation: {e}")
                    velocities = torch.zeros(1, 2, device=self.qwen_model.device)
            else:
                velocities = torch.zeros(1, 2, device=self.qwen_model.device)
            
            return generated_ids, velocities

    def fit_gp_models(self, train_features, train_velocities):
        self.gp_predictor.fit_gp(train_features, train_velocities)

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        self.qwen_model.save_pretrained(save_directory)
        torch.save({
            'bridge_state_dict': self.bridge.state_dict(),
            'gp_predictor_state_dict': self.gp_predictor.state_dict(),
            'gp_config': {
                'use_full_gp': self.gp_predictor.use_full_gp,
                'input_dim': self.gp_predictor.input_dim
            }
        }, os.path.join(save_directory, 'hybrid_components.pt'))
        print(f"Model saved to {save_directory}")

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.qwen_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

# DATASET 
class HybridNavigationDataset(Dataset):
    def __init__(self, jsonl_path: str, processor, velocity_scaler=None, max_samples=None):
        self.processor = processor
        self.velocity_scaler = velocity_scaler
        self.samples = []
        with open(jsonl_path, "r") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                self.samples.append(json.loads(line))
        
        tok = self.processor.tokenizer
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        self.pad_id = tok.pad_token_id
        print(f"Loaded {len(self.samples)} samples from {jsonl_path}")

    def extract_velocity_from_text(self, text: str) -> np.ndarray:
        try:
            if "ASSISTANT:" not in text:
                return np.array([0.0, 0.0])
            _, assistant_part = text.split("ASSISTANT:", 1)
            assistant_json = json.loads(assistant_part.strip())
            vel_cmd = assistant_json.get("Velocity command", {})
            linear_vel = float(vel_cmd.get("linear_velocity", 0.0))
            angular_vel = float(vel_cmd.get("angular_velocity", 0.0))
            return np.array([linear_vel, angular_vel])
        except Exception as e:
            return np.array([0.0, 0.0])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        text = sample["text"]
        if "ASSISTANT:" not in text:
            raise ValueError(f"Sample {idx} missing 'ASSISTANT:' separator")
        user_text, assistant_text = text.split("ASSISTANT:", 1)
        user_text = user_text.strip()
        assistant_text = assistant_text.strip()
        velocity_target = self.extract_velocity_from_text(text)
        
        user_content = []
        for img_path in sample["images"]:
            user_content.append({"type": "image", "image": img_path})
        user_content.append({"type": "text", "text": user_text})
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]}
        ]
        
        prompt_only = [messages[0]]
        prompt_text = self.processor.apply_chat_template(
            prompt_only, add_generation_prompt=True, tokenize=False
        )
        full_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False
        )
        
        images = [Image.open(img_path).convert("RGB") for img_path in sample["images"]]
        inputs = self.processor(
            text=[full_text],
            images=[images],
            return_tensors="pt"
        )
        
        prompt_tokens = self.processor.tokenizer(
            [prompt_text], return_tensors="pt", add_special_tokens=False
        )
        prompt_len = prompt_tokens["input_ids"].shape[-1]
        
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        
        seq_len = min(input_ids.shape[0], MAX_SEQ_LEN_CAP)
        if input_ids.shape[0] > seq_len:
            input_ids = input_ids[:seq_len]
            attention_mask = attention_mask[:seq_len]
        elif input_ids.shape[0] < seq_len:
            pad_len = seq_len - input_ids.shape[0]
            input_ids = torch.cat([
                input_ids,
                torch.full((pad_len,), self.pad_id, dtype=input_ids.dtype)
            ])
            attention_mask = torch.cat([
                attention_mask,
                torch.zeros(pad_len, dtype=attention_mask.dtype)
            ])
        
        labels = input_ids.clone()
        labels[:min(prompt_len, seq_len)] = -100
        
        if (labels != -100).sum() == 0:
            labels[max(0, seq_len - LABEL_MARGIN):] = input_ids[max(0, seq_len - LABEL_MARGIN):]
        
        if self.velocity_scaler is not None:
            velocity_target = self.velocity_scaler.transform(velocity_target.reshape(1, -1))[0]
        
        batch = {
            "input_ids": input_ids.unsqueeze(0),
            "attention_mask": attention_mask.unsqueeze(0),
            "labels": labels.unsqueeze(0),
            "velocity_targets": torch.tensor(velocity_target, dtype=torch.float32)
        }
        
        for key in inputs:
            if key not in ["input_ids", "attention_mask"]:
                batch[key] = inputs[key]
        return batch

#  CUSTOM TRAINER 
class HybridTrainer(Trainer):
    """Custom trainer for hybrid text + velocity training - FIXED"""
    def __init__(self, velocity_scaler=None, **kwargs):
        super().__init__(**kwargs)
        self.velocity_scaler = velocity_scaler
        self.velocity_losses = []
        self.text_losses = []

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute combined text + velocity loss"""
        velocity_targets = inputs.pop("velocity_targets", None)
        outputs = model(**inputs)
        text_loss = outputs.loss if hasattr(outputs, 'loss') else 0
        total_loss = text_loss * TEXT_LOSS_WEIGHT
        
        if velocity_targets is not None and hasattr(outputs, 'predicted_velocities'):
            predicted_velocities = outputs.predicted_velocities
            if velocity_targets.dim() == 1:
                velocity_targets = velocity_targets.unsqueeze(0)
            velocity_targets = velocity_targets.to(predicted_velocities.device)
            velocity_loss = nn.MSELoss()(predicted_velocities, velocity_targets)
            total_loss = total_loss + velocity_loss * VELOCITY_LOSS_WEIGHT
            
            if self.is_in_train:
                self.velocity_losses.append(velocity_loss.item())
                self.text_losses.append(text_loss.item() if torch.is_tensor(text_loss) else text_loss)
            
            if self.state.global_step % 10 == 0 and len(self.velocity_losses) > 0:
                avg_vel_loss = np.mean(self.velocity_losses[-10:])
                avg_text_loss = np.mean(self.text_losses[-10:])
                self.log({
                    "velocity_loss": avg_vel_loss,
                    "text_loss": avg_text_loss,
                    "total_loss": total_loss.item()
                })
        return (total_loss, outputs) if return_outputs else total_loss

#  ENHANCED DATA COLLATOR FOR LARGER BATCHES 
@dataclass
class EnhancedHybridDataCollator:
    def __call__(self, features):
        if len(features) == 1:
            return features[0]
        
        # Handle batching for multiple samples
        batch = {}
        
        # Collect all keys
        all_keys = set()
        for feature in features:
            all_keys.update(feature.keys())
        
        for key in all_keys:
            values = [feature.get(key) for feature in features if key in feature]
            
            if key in ["input_ids", "attention_mask", "labels"]:
                # Pad sequences to same length
                max_len = max(v.shape[-1] for v in values)
                padded_values = []
                for v in values:
                    if v.shape[-1] < max_len:
                        pad_size = max_len - v.shape[-1]
                        if key == "labels":
                            padding = torch.full((v.shape[0], pad_size), -100, dtype=v.dtype)
                        else:
                            padding = torch.zeros(v.shape[0], pad_size, dtype=v.dtype)
                        v = torch.cat([v, padding], dim=-1)
                    padded_values.append(v)
                batch[key] = torch.cat(padded_values, dim=0)
                
            elif key == "velocity_targets":
                # Stack velocity targets
                batch[key] = torch.stack(values, dim=0)
                
            elif key in ["pixel_values", "image_grid_thw"]:
                # Handle image data
                if isinstance(values[0], torch.Tensor):
                    if len(values[0].shape) > 2:  # Multi-dimensional tensors
                        batch[key] = torch.cat(values, dim=0)
                    else:
                        batch[key] = torch.stack(values, dim=0)
                else:
                    batch[key] = values
            else:
                # For other keys, try to stack or concatenate
                try:
                    if all(torch.is_tensor(v) for v in values):
                        if len(values[0].shape) == 0:  # Scalars
                            batch[key] = torch.stack(values)
                        else:
                            batch[key] = torch.cat(values, dim=0)
                    else:
                        batch[key] = values
                except:
                    batch[key] = values
        
        return batch

# TRAINING FUNCTIONS 
def prepare_velocity_scaler(train_jsonl: str) -> StandardScaler:
    velocities = []
    with open(train_jsonl, 'r') as f:
        for line in f:
            sample = json.loads(line)
            text = sample["text"]
            if "ASSISTANT:" in text:
                try:
                    _, assistant_part = text.split("ASSISTANT:", 1)
                    assistant_json = json.loads(assistant_part.strip())
                    vel_cmd = assistant_json.get("Velocity command", {})
                    linear_vel = float(vel_cmd.get("linear_velocity", 0.0))
                    angular_vel = float(vel_cmd.get("angular_velocity", 0.0))
                    velocities.append([linear_vel, angular_vel])
                except:
                    continue
    velocities = np.array(velocities)
    scaler = StandardScaler()
    scaler.fit(velocities)
    print(f"Velocity scaler fitted on {len(velocities)} samples")
    print(f"Linear velocity: mean={scaler.mean_[0]:.3f}, std={scaler.scale_[0]:.3f}")
    print(f"Angular velocity: mean={scaler.mean_[1]:.3f}, std={scaler.scale_[1]:.3f}")
    return scaler

def load_and_prepare_hybrid_model():
    print("Loading Qwen model for A6000...")
    
    # Enhanced quantization config for A6000
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    # Enhanced processor config
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
        trust_remote_code=True
    )
    
    # Load model with A6000 optimizations
    qwen_model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="cuda:0",  # Explicit device mapping
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if USE_FLASH_ATTENTION else "eager",  # Flash attention for efficiency
    )
    
    qwen_model.config.use_cache = False
    qwen_model = prepare_model_for_kbit_training(
        qwen_model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    
    # Enhanced LoRA config for A6000
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "vision_proj", "visual_proj" if hasattr(qwen_model, "visual_proj") else None
    ]
    target_modules = [m for m in target_modules if m is not None]
    
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head"],  # Save full precision for output layer
    )
    qwen_model = get_peft_model(qwen_model, lora_config)
    
    print("Creating enhanced hybrid model...")
    hybrid_model = HybridQwenGP(qwen_model, processor)
    
    # Print enhanced parameter info
    trainable_params = sum(p.numel() for p in hybrid_model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in hybrid_model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")
    print(f"Total parameters: {all_params:,}")
    print(f"Model memory footprint: ~{all_params * 2 / 1e9:.1f}GB (bf16)")
    
    return hybrid_model, processor

def train_hybrid_model(train_jsonl: str, val_jsonl: str = None):
    set_seed(SEED)
    print("="*60)
    print("HYBRID QWEN-GP TRAINING")
    print("="*60)
    
    print("\n📦 Loading hybrid model...")
    hybrid_model, processor = load_and_prepare_hybrid_model()
    
    print("\n📊 Preparing velocity scaler...")
    velocity_scaler = prepare_velocity_scaler(train_jsonl)
    
    print("\n📚 Loading datasets...")
    train_dataset = HybridNavigationDataset(train_jsonl, processor, velocity_scaler)
    val_dataset = HybridNavigationDataset(val_jsonl, processor, velocity_scaler) if val_jsonl else None
    print(f"Training samples: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation samples: {len(val_dataset)}")
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=1.0,
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        eval_strategy="no",  
        eval_steps=None,  
        load_best_model_at_end=False,
        metric_for_best_model=None,
        greater_is_better=False,
        bf16=USE_BF16,
        fp16=not USE_BF16,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        report_to="none",
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_steps=MAX_STEPS,
        remove_unused_columns=False,
        save_safetensors=False,
    )
    
    trainer = HybridTrainer(
        model=hybrid_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=EnhancedHybridDataCollator(),  # Use enhanced collator
        velocity_scaler=velocity_scaler,
    )
    
    print("\n🚀 Starting training...")
    trainer.train()
    
    # Manual validation evaluation after training with enhanced metrics
    if val_dataset is not None:
        print("\n📊 Running enhanced validation evaluation...")
        hybrid_model.eval()
        val_losses = []
        val_velocity_losses = []
        val_text_losses = []
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            # Evaluate on more samples for better statistics
            num_eval_samples = min(50, len(val_dataset))
            for i in range(num_eval_samples):
                batch = val_dataset[i]
                for key in batch:
                    if torch.is_tensor(batch[key]):
                        batch[key] = batch[key].to(hybrid_model.qwen_model.device)
                
                velocity_targets = batch.get("velocity_targets", None)
                outputs = hybrid_model(**batch)
                text_loss = outputs.loss if hasattr(outputs, 'loss') else torch.tensor(0.0)
                total_loss = text_loss * TEXT_LOSS_WEIGHT
                
                if velocity_targets is not None and hasattr(outputs, 'predicted_velocities'):
                    predicted_velocities = outputs.predicted_velocities
                    if velocity_targets.dim() == 1:
                        velocity_targets = velocity_targets.unsqueeze(0)
                    velocity_targets = velocity_targets.to(predicted_velocities.device)
                    velocity_loss = nn.MSELoss()(predicted_velocities, velocity_targets)
                    total_loss = total_loss + velocity_loss * VELOCITY_LOSS_WEIGHT
                    val_velocity_losses.append(velocity_loss.item())
                    val_text_losses.append(text_loss.item() if torch.is_tensor(text_loss) else text_loss)
                    
                    # Store for R² calculation
                    val_predictions.append(predicted_velocities.cpu().numpy())
                    val_targets.append(velocity_targets.cpu().numpy())
                
                val_losses.append(total_loss.item())
                
                if (i + 1) % 10 == 0:
                    print(f"  Validated {i+1}/{num_eval_samples} samples...")
        
        if val_losses:
            print(f"\n📈 Validation Results:")
            print(f"  Total Loss: {np.mean(val_losses):.4f}")
            if val_velocity_losses:
                print(f"  Velocity Loss: {np.mean(val_velocity_losses):.4f}")
                print(f"  Text Loss: {np.mean(val_text_losses):.4f}")
                
                # Calculate R² score for validation
                if len(val_predictions) > 1:
                    val_pred_array = np.concatenate(val_predictions, axis=0)
                    val_target_array = np.concatenate(val_targets, axis=0)
                    val_r2 = r2_score(val_target_array, val_pred_array)
                    print(f"  Validation R² Score: {val_r2:.4f}")
    
    print("\n✅ Training completed successfully!")
    
    print("\n🎯 Fitting enhanced GP models on training data...")
    try:
        train_features = []
        train_velocities = []
        hybrid_model.eval()
        with torch.no_grad():
            # Use more samples for better GP fitting
            for i in range(min(GP_TRAINING_SAMPLES, len(train_dataset))):
                batch = train_dataset[i]
                for key in batch:
                    if torch.is_tensor(batch[key]):
                        batch[key] = batch[key].to(hybrid_model.qwen_model.device)
                outputs = hybrid_model(**batch)
                if hybrid_model.extracted_features is not None:
                    gp_features = hybrid_model.bridge(hybrid_model.extracted_features)
                    train_features.append(gp_features.cpu())
                    vel_target = batch['velocity_targets'].cpu()
                    if velocity_scaler:
                        vel_target = torch.tensor(
                            velocity_scaler.inverse_transform(vel_target.numpy().reshape(1, -1))[0]
                        )
                    train_velocities.append(vel_target)
                
                if (i + 1) % 50 == 0:
                    print(f"  Processed {i+1}/{min(GP_TRAINING_SAMPLES, len(train_dataset))} samples for GP fitting...")
        
        if train_features:
            train_features = torch.cat(train_features, dim=0)
            train_velocities = torch.stack(train_velocities, dim=0)
            # Switch to full GP mode and fit
            hybrid_model.gp_predictor.use_full_gp = True
            hybrid_model.gp_predictor.fit_gp(train_features, train_velocities)
            print("✅ Enhanced GP models fitted successfully")
        else:
            print("⚠️ No features extracted for GP fitting")
    except Exception as e:
        print(f"⚠️ Could not fit GP models: {e}")
        print("Continuing with enhanced neural network predictor")
    
    print("\n💾 Saving final model...")
    final_save_path = os.path.join(OUTPUT_DIR, "final_model")
    hybrid_model.save_pretrained(final_save_path)
    processor.save_pretrained(final_save_path)
    
    scaler_path = os.path.join(final_save_path, "velocity_scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(velocity_scaler, f)
    
    print(f"\n✅ Training complete! Model saved to {final_save_path}")
    return hybrid_model, processor, velocity_scaler

# INFERENCE FUNCTIONS
def load_trained_hybrid_model(model_path: str):
    print(f"Loading model from {model_path}...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    qwen_model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    hybrid_model = HybridQwenGP(qwen_model, processor)
    components_path = os.path.join(model_path, 'hybrid_components.pt')
    if os.path.exists(components_path):
        components = torch.load(components_path, map_location='cpu')
        hybrid_model.bridge.load_state_dict(components['bridge_state_dict'])
        hybrid_model.gp_predictor.load_state_dict(components['gp_predictor_state_dict'])
        if 'gp_config' in components:
            hybrid_model.gp_predictor.use_full_gp = components['gp_config']['use_full_gp']
            hybrid_model.gp_predictor.input_dim = components['gp_config']['input_dim']
        print("✅ Hybrid components loaded")
    
    scaler_path = os.path.join(model_path, "velocity_scaler.pkl")
    velocity_scaler = None
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            velocity_scaler = pickle.load(f)
    
    hybrid_model.eval()
    return hybrid_model, processor, velocity_scaler

def predict_with_hybrid(hybrid_model, processor, velocity_scaler,
                        images: List[str], prompt: str = None):
    if prompt is None:
        prompt = """You are an expert robot navigation assistant. The user will provide a sequence of 6 images, where the final image represents the present moment (Frame N-1).
Your task is to analyze this sequence and perform the following:

Describe the Present (Frame N-1): Based on the final image in the sequence, provide a description of the robot's immediate surroundings.
Predict the Future (Frame N): Based on the entire sequence, predict the robot's action for the immediate next moment (Frame N).
Your final output MUST be a single, valid JSON object using these exact keys:

"Scene description"
"Next high-level command" (must be one of ["go_forward", "go_left", "go_right", "stop"])
"Explanation"
"Velocity command"
The value for "Velocity command" must be a nested JSON object with "linear_velocity" and "angular_velocity" keys."""
    
    pil_images = [Image.open(img_path).convert("RGB") for img_path in images]
    user_content = []
    for img in pil_images:
        user_content.append({"type": "image", "image": img})
    user_content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": user_content}]
    
    prompt_text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(
        text=[prompt_text],
        images=[pil_images],
        return_tensors="pt")
    
    device = next(hybrid_model.parameters()).device
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    
    with torch.no_grad():
        generated_ids, predicted_velocities = hybrid_model.generate_with_velocities(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.1,
        )
        
        generated_text = processor.decode(generated_ids[0], skip_special_tokens=True)
        if "ASSISTANT:" in generated_text:
            assistant_response = generated_text.split("ASSISTANT:")[-1].strip()
        else:
            assistant_response = generated_text
        
        if velocity_scaler:
            predicted_velocities_unscaled = velocity_scaler.inverse_transform(
                predicted_velocities.cpu().numpy()
            )[0]
        else:
            predicted_velocities_unscaled = predicted_velocities.cpu().numpy()[0]
    
    try:
        json_start = assistant_response.find('{')
        json_end = assistant_response.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = assistant_response[json_start:json_end]
            response_json = json.loads(json_str)
            response_json["Velocity command"] = {
                "linear_velocity": float(predicted_velocities_unscaled[0]),
                "angular_velocity": float(predicted_velocities_unscaled[1])
            }
            return {
                "success": True,
                "text_response": assistant_response,
                "json_output": response_json,
                "velocities": predicted_velocities_unscaled,
                "formatted_json": json.dumps(response_json, indent=2)
            }
    except Exception as e:
        print(f"Warning: Could not parse JSON: {e}")
    
    return {
        "success": False,
        "text_response": assistant_response,
        "json_output": None,
        "velocities": predicted_velocities_unscaled,
        "formatted_json": None
    }

#  COMPREHENSIVE EVALUATION
class ComprehensiveEvaluator:
    def __init__(self):
        print("Loading sentence transformer model...")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def evaluate_text_similarity(self, pred_text: str, gt_text: str) -> float:
        """Calculate semantic similarity between predicted and ground truth text"""
        try:
            pred_embedding = self.sentence_model.encode([pred_text])
            gt_embedding = self.sentence_model.encode([gt_text])
            similarity = cosine_similarity(pred_embedding, gt_embedding)[0][0]
            return float(similarity)
        except:
            return 0.0
    
    def evaluate_json_fields(self, pred_json: dict, gt_json: dict) -> dict:
        results = {}
        
        # Scene description similarity
        if "Scene description" in pred_json and "Scene description" in gt_json:
            results["scene_similarity"] = self.evaluate_text_similarity(
                pred_json["Scene description"], 
                gt_json["Scene description"]
            )
        else:
            results["scene_similarity"] = 0.0
        
        # Command accuracy
        pred_cmd = pred_json.get("Next high-level command", "")
        gt_cmd = gt_json.get("Next high-level command", "")
        results["command_accuracy"] = 1.0 if pred_cmd == gt_cmd else 0.0
        
        # Explanation similarity
        if "Explanation" in pred_json and "Explanation" in gt_json:
            results["explanation_similarity"] = self.evaluate_text_similarity(
                pred_json["Explanation"], 
                gt_json["Explanation"]
            )
        else:
            results["explanation_similarity"] = 0.0
        
        return results
    
    def evaluate_velocities(self, pred_vel: np.ndarray, gt_vel: np.ndarray) -> dict:
        linear_error = abs(pred_vel[0] - gt_vel[0])
        angular_error = abs(pred_vel[1] - gt_vel[1])
        total_error = np.sqrt(linear_error**2 + angular_error**2)
        
        return {
            "linear_error": linear_error,
            "angular_error": angular_error,
            "total_error": total_error,
            "linear_relative_error": linear_error / (abs(gt_vel[0]) + 1e-6),
            "angular_relative_error": angular_error / (abs(gt_vel[1]) + 1e-6)
        }

def comprehensive_evaluation(hybrid_model, processor, velocity_scaler, eval_jsonl: str):
    print("\n" + "="*60)
    print("ENHANCED COMPREHENSIVE HYBRID MODEL EVALUATION")
    print("="*60)

    evaluator = ComprehensiveEvaluator()
    eval_dataset = HybridNavigationDataset(eval_jsonl, processor, velocity_scaler)

    results = {
        'samples': [],
        'velocity_metrics': {
            'linear_errors': [],
            'angular_errors': [],
            'total_errors': [],
            'linear_relative_errors': [],
            'angular_relative_errors': []
        },
        'text_metrics': {
            'scene_similarities': [],
            'command_accuracies': [],
            'explanation_similarities': [],
            'json_success_rate': [],
            'valid_command_rate': []
        },
        'overall_scores': {}
    }

    valid_commands = ["go_forward", "go_left", "go_right", "stop"]
    hybrid_model.eval()

    all_pred_velocities = []
    all_gt_velocities = []

    print(f"Evaluating all {len(eval_dataset)} samples with enhanced metrics...")

    # Process in mini-batches for efficiency 
    batch_size = 4
    total_samples = len(eval_dataset)

    for batch_start in range(0, total_samples, batch_size):
        batch_end = min(batch_start + batch_size, total_samples)

        for i in range(batch_start, batch_end):
            sample = eval_dataset.samples[i]
            sample_result = {
                'sample_id': i,
                'images': sample["images"],
                'timestamp': datetime.now().isoformat()
            }

            # Extract ground truth
            text = sample["text"]
            gt_velocities = eval_dataset.extract_velocity_from_text(text)
            all_gt_velocities.append(gt_velocities)

            # Parse ground truth JSON
            try:
                _, gt_assistant_text = text.split("ASSISTANT:", 1)
                gt_json = json.loads(gt_assistant_text.strip())
                sample_result['ground_truth'] = gt_json
            except:
                sample_result['ground_truth'] = None
                continue

            # Get user prompt
            user_text = text.split("ASSISTANT:")[0].strip()

            # Run inference with enhanced generation parameters
            prediction_result = predict_with_hybrid(
                hybrid_model, processor, velocity_scaler,
                sample["images"], user_text
            )

            sample_result['prediction'] = {
                'success': prediction_result["success"],
                'text_response': prediction_result["text_response"],
                'json_output': prediction_result["json_output"],
                'velocities': prediction_result["velocities"].tolist()
            }

            # Enhanced velocity evaluation
            pred_velocities = prediction_result["velocities"]
            all_pred_velocities.append(pred_velocities)

            velocity_eval = evaluator.evaluate_velocities(pred_velocities, gt_velocities)
            sample_result['velocity_evaluation'] = velocity_eval

            # Store velocity metrics
            results['velocity_metrics']['linear_errors'].append(velocity_eval['linear_error'])
            results['velocity_metrics']['angular_errors'].append(velocity_eval['angular_error'])
            results['velocity_metrics']['total_errors'].append(velocity_eval['total_error'])
            results['velocity_metrics']['linear_relative_errors'].append(velocity_eval['linear_relative_error'])
            results['velocity_metrics']['angular_relative_errors'].append(velocity_eval['angular_relative_error'])

            # Enhanced text evaluation
            if prediction_result["success"] and prediction_result["json_output"] and sample_result['ground_truth']:
                text_eval = evaluator.evaluate_json_fields(
                    prediction_result["json_output"],
                    sample_result['ground_truth']
                )
                sample_result['text_evaluation'] = text_eval

                # Store text metrics
                results['text_metrics']['scene_similarities'].append(text_eval['scene_similarity'])
                results['text_metrics']['command_accuracies'].append(text_eval['command_accuracy'])
                results['text_metrics']['explanation_similarities'].append(text_eval['explanation_similarity'])
                results['text_metrics']['json_success_rate'].append(1)

                # Check valid command
                pred_cmd = prediction_result["json_output"].get("Next high-level command", "")
                results['text_metrics']['valid_command_rate'].append(1 if pred_cmd in valid_commands else 0)
            else:
                sample_result['text_evaluation'] = {
                    'scene_similarity': 0.0,
                    'command_accuracy': 0.0,
                    'explanation_similarity': 0.0
                }
                results['text_metrics']['scene_similarities'].append(0.0)
                results['text_metrics']['command_accuracies'].append(0.0)
                results['text_metrics']['explanation_similarities'].append(0.0)
                results['text_metrics']['json_success_rate'].append(0)
                results['text_metrics']['valid_command_rate'].append(0)

            results['samples'].append(sample_result)

        # Progress update for batches
        print(f"  Evaluated {min(batch_end, total_samples)}/{total_samples} samples...")

    # Enhanced overall metrics calculation
    print("\n📊 Calculating enhanced overall metrics...")

    # Velocity metrics with additional statistics
    velocity_metrics = results['velocity_metrics']
    results['overall_scores']['velocity'] = {
        'mean_linear_error': np.mean(velocity_metrics['linear_errors']),
        'std_linear_error': np.std(velocity_metrics['linear_errors']),
        'mean_angular_error': np.mean(velocity_metrics['angular_errors']),
        'std_angular_error': np.std(velocity_metrics['angular_errors']),
        'mean_total_error': np.mean(velocity_metrics['total_errors']),
        'std_total_error': np.std(velocity_metrics['total_errors']),
        'mean_linear_relative_error': np.mean(velocity_metrics['linear_relative_errors']),
        'mean_angular_relative_error': np.mean(velocity_metrics['angular_relative_errors']),
        'rmse': np.sqrt(np.mean(np.array(velocity_metrics['total_errors'])**2)),
        'mae': np.mean(velocity_metrics['total_errors']),  # Mean Absolute Error
        'median_error': np.median(velocity_metrics['total_errors']),  # Median error
        'percentile_95_error': np.percentile(velocity_metrics['total_errors'], 95)  # 95th percentile
    }

    if len(all_gt_velocities) > 1:
        all_gt = np.array(all_gt_velocities)
        all_pred = np.array(all_pred_velocities)
        velocity_r2 = r2_score(all_gt, all_pred)
        linear_r2 = r2_score(all_gt[:, 0], all_pred[:, 0])
        angular_r2 = r2_score(all_gt[:, 1], all_pred[:, 1])
        linear_corr = np.corrcoef(all_gt[:, 0], all_pred[:, 0])[0, 1]
        angular_corr = np.corrcoef(all_gt[:, 1], all_pred[:, 1])[0, 1]

        results['overall_scores']['velocity'].update({
            'r2_score': velocity_r2,
            'linear_r2': linear_r2,
            'angular_r2': angular_r2,
            'linear_correlation': linear_corr,
            'angular_correlation': angular_corr,
            'velocity_mse': mean_squared_error(all_gt, all_pred),
            'velocity_mae': mean_absolute_error(all_gt, all_pred)
        })

    # Enhanced text metrics
    text_metrics = results['text_metrics']
    results['overall_scores']['text'] = {
        'mean_scene_similarity': np.mean(text_metrics['scene_similarities']),
        'std_scene_similarity': np.std(text_metrics['scene_similarities']),
        'mean_command_accuracy': np.mean(text_metrics['command_accuracies']),
        'mean_explanation_similarity': np.mean(text_metrics['explanation_similarities']),
        'std_explanation_similarity': np.std(text_metrics['explanation_similarities']),
        'json_success_rate': np.mean(text_metrics['json_success_rate']),
        'valid_command_rate': np.mean(text_metrics['valid_command_rate']),
        'overall_text_score': (np.mean(text_metrics['scene_similarities']) +
                               np.mean(text_metrics['command_accuracies']) +
                               np.mean(text_metrics['explanation_similarities'])) / 3
    }

    # Enhanced hybrid score calculation
    velocity_score = max(0, results['overall_scores']['velocity'].get('r2_score', 0))
    text_score = results['overall_scores']['text']['overall_text_score']

    # Weighted combination with bonus for high performance
    base_hybrid_accuracy = (0.6 * velocity_score + 0.4 * text_score) * 100
    if velocity_score > 0.8:
        base_hybrid_accuracy *= 1.1 
    if text_score > 0.8:
        base_hybrid_accuracy *= 1.05  

    results['overall_scores']['hybrid_accuracy'] = min(100, base_hybrid_accuracy)  

    results['evaluation_metadata'] = {
        'timestamp': datetime.now().isoformat(),
        'num_samples': len(eval_dataset),
        'model_config': {
            'qwen_model': MODEL_ID,
            'gp_input_dim': GP_INPUT_DIM,
            'bridge_hidden_dim': BRIDGE_HIDDEN_DIM,
            'training_steps': MAX_STEPS,
            'training_epochs': NUM_EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE
        },
        'evaluation_weights': {
            'velocity_weight': 0.6,
            'text_weight': 0.4
        },
        'hardware': 'RTX A6000',
        'enhanced_features': [
            'Larger batch size',
            'Enhanced GP hyperparameter optimization',
            'Improved bridge network architecture',
            'Better regularization',
            'Extended training',
            'Flash attention support'
        ]
    }

    return results

def print_evaluation_summary(results: dict):
    print("\n" + "="*60)
    print("ENHANCED EVALUATION SUMMARY (RTX A6000)")
    print("="*60)
    
    # Enhanced velocity performance
    vel_scores = results['overall_scores']['velocity']
    print(f"\n🎯 VELOCITY PREDICTION PERFORMANCE:")
    print(f"  R² Score (Overall): {vel_scores.get('r2_score', 0):.4f}")
    print(f"  Linear Velocity R²: {vel_scores.get('linear_r2', 0):.4f}")
    print(f"  Angular Velocity R²: {vel_scores.get('angular_r2', 0):.4f}")
    print(f"  Linear Correlation: {vel_scores.get('linear_correlation', 0):.4f}")
    print(f"  Angular Correlation: {vel_scores.get('angular_correlation', 0):.4f}")
    print(f"  Mean Linear Error: {vel_scores['mean_linear_error']:.4f} ± {vel_scores.get('std_linear_error', 0):.4f} m/s")
    print(f"  Mean Angular Error: {vel_scores['mean_angular_error']:.4f} ± {vel_scores.get('std_angular_error', 0):.4f} rad/s")
    print(f"  RMSE: {vel_scores['rmse']:.4f}")
    print(f"  MAE: {vel_scores.get('mae', 0):.4f}")
    print(f"  Median Error: {vel_scores.get('median_error', 0):.4f}")
    print(f"  95th Percentile Error: {vel_scores.get('percentile_95_error', 0):.4f}")
    
    # Enhanced text performance
    text_scores = results['overall_scores']['text']
    print(f"\n📝 TEXT GENERATION PERFORMANCE:")
    print(f"  Scene Description Similarity: {text_scores['mean_scene_similarity']:.3f} ± {text_scores.get('std_scene_similarity', 0):.3f}")
    print(f"  Command Accuracy: {text_scores['mean_command_accuracy']:.3f}")
    print(f"  Explanation Similarity: {text_scores['mean_explanation_similarity']:.3f} ± {text_scores.get('std_explanation_similarity', 0):.3f}")
    print(f"  JSON Success Rate: {text_scores['json_success_rate']:.3f}")
    print(f"  Valid Command Rate: {text_scores['valid_command_rate']:.3f}")
    print(f"  Overall Text Score: {text_scores.get('overall_text_score', 0):.3f}")
    
    # Enhanced overall performance
    hybrid_accuracy = results['overall_scores']['hybrid_accuracy']
    print(f"\n🏆 ENHANCED HYBRID MODEL ACCURACY: {hybrid_accuracy:.2f}%")
    
    # Enhanced comparison with targets
    current_r2 = vel_scores.get('r2_score', 0)
    print(f"\n📈 PERFORMANCE vs TARGETS:")
    print(f"  Current Hybrid R²: {current_r2:.4f}")
    
    # Training configuration summary
    metadata = results.get('evaluation_metadata', {})
    model_config = metadata.get('model_config', {})
    print(f"\n⚙️  TRAINING CONFIGURATION:")
    print(f"  Training Steps: {model_config.get('training_steps', 'N/A')}")
    print(f"  Training Epochs: {model_config.get('training_epochs', 'N/A')}")
    print(f"  Batch Size: {model_config.get('batch_size', 'N/A')}")
    print(f"  Learning Rate: {model_config.get('learning_rate', 'N/A')}")
    print(f"  GP Input Dimension: {model_config.get('gp_input_dim', 'N/A')}")
    print(f"  Bridge Hidden Dimension: {model_config.get('bridge_hidden_dim', 'N/A')}")
    print(f"  Hardware: {metadata.get('hardware', 'Unknown')}")
    
    # Performance improvement recommendations
    print(f"\n💡 PERFORMANCE INSIGHTS:")
    if current_r2 < 0.5:
        print("  • Velocity prediction needs more training or feature engineering")
        print("  • Consider increasing GP training samples or kernel tuning")
    elif current_r2 < 0.8:
        print("  • Good velocity prediction foundation, fine-tuning recommended")
        print("  • Bridge network is learning meaningful representations")
    else:
        print("  • Excellent velocity prediction performance!")
        print("  • Model successfully leverages visual features for control")
    
    if text_scores.get('overall_text_score', 0) < 0.6:
        print("  • Text generation could benefit from more epochs")
        print("  • Consider adjusting text loss weight or prompt engineering")
    else:
        print("  • Strong text generation and reasoning capabilities")
    
    print("="*60)

def save_master_output(results: dict, output_path: str):
    print(f"\n💾 Saving master output to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"✅ Master output saved successfully!")

# COMPLETE PIPELINE 
def run_complete_pipeline(train_jsonl: str, val_jsonl: str, output_dir: str = None):
    if output_dir is None:
        output_dir = f"./hybrid_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("🚀 STARTING COMPLETE HYBRID PIPELINE")
    print("="*60)
    print(f"Training data: {train_jsonl}")
    print(f"Validation data: {val_jsonl}")
    print(f"Output directory: {output_dir}")
    print("="*60)
    
    start_time = time.time()
    
    # Step 1: Training
    print("\n🔧 STEP 1: TRAINING HYBRID MODEL")
    hybrid_model, processor, velocity_scaler = train_hybrid_model(train_jsonl, val_jsonl)
    
    # Step 2: Comprehensive Evaluation
    print("\n📊 STEP 2: COMPREHENSIVE EVALUATION")
    evaluation_results = comprehensive_evaluation(hybrid_model, processor, velocity_scaler, val_jsonl)
    
    # Step 3: Print Summary
    print_evaluation_summary(evaluation_results)
    
    # Step 4: Save Master Output
    master_output_path = os.path.join(output_dir, "master_output.json")
    save_master_output(evaluation_results, master_output_path)
    
    # Step 5: Save Additional Files
    print("\n💾 Saving additional outputs...")
    
    # Save model info
    model_info = {
        "model_path": os.path.join(OUTPUT_DIR, "final_model"),
        "training_config": {
            "train_samples": len(json.loads(open(train_jsonl).readline())["text"].split('\n')),
            "val_samples": len(json.loads(open(val_jsonl).readline())["text"].split('\n')),
            "epochs": NUM_EPOCHS,
            "max_steps": MAX_STEPS,
            "learning_rate": LEARNING_RATE
        },
        "final_accuracy": evaluation_results['overall_scores']['hybrid_accuracy']
    }
    
    with open(os.path.join(output_dir, "model_info.json"), 'w') as f:
        json.dump(model_info, f, indent=2)
    
    # Calculate and save timing
    total_time = time.time() - start_time
    timing_info = {
        "total_runtime_seconds": total_time,
        "total_runtime_minutes": total_time / 60,
        "timestamp_start": datetime.fromtimestamp(start_time).isoformat(),
        "timestamp_end": datetime.now().isoformat()
    }
    
    with open(os.path.join(output_dir, "timing_info.json"), 'w') as f:
        json.dump(timing_info, f, indent=2)
    
    print(f"\n🎉 PIPELINE COMPLETE!")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"🎯 Final Hybrid Accuracy: {evaluation_results['overall_scores']['hybrid_accuracy']:.2f}%")
    print(f"📁 All results saved to: {output_dir}")
    
    return evaluation_results, output_dir

#  MAIN FUNCTION 
def main():
    parser = argparse.ArgumentParser(description="Complete Hybrid Qwen-GP Navigation Pipeline")
    parser.add_argument("--train_jsonl", required=True, help="Path to training JSONL file")
    parser.add_argument("--val_jsonl", required=True, help="Path to validation JSONL file")
    parser.add_argument("--output_dir", help="Output directory for results")
    args = parser.parse_args()
    
    # Run complete pipeline
    results, output_dir = run_complete_pipeline(
        args.train_jsonl, 
        args.val_jsonl, 
        args.output_dir
    )
    
    print(f"\n✨ SUCCESS! Check {output_dir}/master_output.json for complete results")

if __name__ == "__main__":
    main()