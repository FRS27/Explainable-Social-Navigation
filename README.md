# Explainable Intent Recognition using Multimodal AI


A novel multimodal deep learning system for socially-aware robot navigation that predicts velocity commands while generating human-understandable explanations of robot behavior.

img width="1159" height="318" alt="image" src="https://github.com/user-attachments/assets/f35a429d-5592-4a98-ae22-e97693ef6b30

*6-frame sequential analysis for robot navigation decisions*

---

## 🎯 Project Overview

This research enables mobile robots to navigate crowded human environments with genuine social awareness—moving beyond simple obstacle avoidance to understanding human intent and social context.

**Core Innovation:** Integration of vision-language models and specialized multimodal architectures for explainable robot control, combining semantic understanding with precise motor prediction.

---

## 🏗️ System Architecture

### Pipeline 1: Dataset Curation with LLMs

Built a comprehensive framework for analyzing 6-frame sequential robot behavior using large language models, evaluating navigation decisions against 8 social navigation principles: safety, comfort, legibility, politeness, social norms, agent understanding, proactivity, and contextual appropriateness.

- Dataset: SCAND (Socially Compliant Navigation Dataset)
- AI-assisted annotation + 100% human verification
- **Annotation Quality:** 0.923-0.941 semantic similarity scores

### Pipeline 2: Vision-Language-Action Model

Fine-tuned **Qwen 2.5-VL (3B parameters)** using **QLoRA** to create a specialized vision-language-action model for dual-task learning.

![Depth Perception](images/depth_maps.png)
*3D spatial understanding using Depth-Anything V2*

**Technical Details:**
- ~100M trainable parameters (LoRA rank 64, alpha 128)
- Hybrid architecture: VLM features → Bridge Network → Gaussian Process
- Simultaneous learning: Semantic text generation + Velocity prediction
- 4-bit quantization with BitsAndBytes

**Text Generation Performance:**
- Scene Description: 0.66 semantic similarity
- Action Justification: 0.66 semantic similarity

### Pipeline 3: Specialized Multimodal Fusion Network

Designed a dual-branch architecture processing temporal sequences through expert feature extractors, achieving precise velocity predictions.

![DINOv2 Attention](images/attention_maps.png)
*Semantic attention mechanisms showing robot focus areas*

**Three Expert Feature Extractors:**
- **DINOv2-base:** 768-dim semantic features
- **Depth-Anything V2:** 256-dim spatial structure (CNN-compressed depth maps)
- **Farnebäck Optical Flow:** 128-dim motion statistics with divergence features

**Architecture Components:**
- Temporal Processing: 2-layer Bidirectional LSTMs (256 hidden units)
- Fusion Mechanism: 8-head Multi-head Self-Attention
- Dual-Branch Design: Separate prediction heads for linear/angular velocity
- Adaptive Bias Correction: Command embeddings with statistical velocity templates

![Optical Flow Analysis](images/optical_flow_comparison.png)
*Optical flow patterns reveal motion characteristics for velocity prediction*

---

## 📊 Results

### Velocity Prediction Performance

| Metric | Angular Velocity | Linear Velocity |
|--------|------------------|-----------------|
| **R² Score** | **0.81** | 0.44 |
| **MAE** | **0.1239** | 0.1391 |

![Prediction Performance](images/prediction_scatter.png)
*Scatter plots showing prediction accuracy for both velocity components*

---

## 🔬 Key Contributions

1. **Integrated Multimodal System:** Combined VLM semantic understanding with specialized geometric feature extraction for robot control
2. **Explainability Suite:** Comprehensive visualization pipeline including attention maps, optical flow analysis, and feature space representations
3. **Dual-Task Learning:** Simultaneous generation of natural language explanations and precise velocity commands
4. **Dataset Innovation:** Gold-standard annotations evaluating social navigation principles with LLM assistance

![Error Analysis](images/error_analysis.png)
*Detailed error distribution analysis across prediction ranges*

---

## 🛠️ Technology Stack

**Deep Learning:** PyTorch, Transformers (Hugging Face), PEFT (LoRA/QLoRA)

**Foundation Models:** Qwen 2.5-VL, DINOv2-base, Depth-Anything V2, Gemini 2.5-Pro

**Computer Vision:** OpenCV (Farnebäck Optical Flow), PIL

**ML Tools:** Scikit-learn, GPyTorch, Sentence Transformers

**Visualization:** Matplotlib, Seaborn, Plotly

---

## 📁 Repository Structure

