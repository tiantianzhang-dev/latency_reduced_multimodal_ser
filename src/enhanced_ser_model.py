"""
Enhanced Speech Emotion Recognition (SER) Model Architecture.

This module implements a multimodal SER model with cross-modal gated interaction (CmGI)
fusion mechanism. The model combines acoustic features from HuBERT and semantic features
from BERT using a temporal-aware gated fusion approach.

Note: This implementation corresponds to the CmGI baseline model, not the proposed
simplified single-layer architecture described in the paper.

Author: Xuefei Bian, Hao-wei Liang, Tiantian Zhang
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import HubertModel, BertModel

class TemporalGatedFusion(nn.Module):
    """
    Temporal-aware gated fusion module.
    
    This module implements a gating mechanism that dynamically balances between
    original features and cross-attended features using a sigmoid gate.
    
    Formula:
        G = sigmoid(W * concat([sa, ca]) + b)
        output = sa * G + ca * (1 - G)
    
    where sa is self-attended features and ca is cross-attended features.
    
    Args:
        dim (int): Feature dimension
    """
    
    def __init__(self, dim):
        super(TemporalGatedFusion, self).__init__()
        self.transform = nn.Linear(dim * 2, dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, sa, ca):
        """
        Apply temporal gated fusion.
        
        Args:
            sa: Self-attended features, shape (batch, time, dim)
            ca: Cross-attended features, shape (batch, time, dim)
            
        Returns:
            Fused features, shape (batch, time, dim)
        """
        concat = torch.cat((sa, ca), dim=-1)
        gate = self.sigmoid(self.transform(concat))
        output = sa * gate + ca * (1 - gate)
        return output

class CmGI(nn.Module):
    """
    Cross-modal Gated Interaction (CmGI) module.
    
    This module implements bidirectional cross-attention between acoustic and semantic
    modalities, followed by temporal-aware gated fusion. This is the baseline model
    from Gao et al. [18].
    
    Architecture:
        1. Project features to query, key, value spaces
        2. Compute bidirectional cross-attention
        3. Apply temporal gated fusion
    
    Args:
        dim (int): Feature dimension (typically 1024)
    """
    
    def __init__(self, dim):
        super(CmGI, self).__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.temporal_fusion = TemporalGatedFusion(dim)

    def attention(self, q, k, v):
        """
        Scaled dot-product attention.
        
        Args:
            q: Query tensor, shape (batch, time, dim)
            k: Key tensor, shape (batch, time, dim)
            v: Value tensor, shape (batch, time, dim)
            
        Returns:
            Attention output, shape (batch, time, dim)
        """
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        attn_probs = torch.softmax(attn_weights, dim=-1)
        return torch.matmul(attn_probs, v)

    def forward(self, xa, xl):
        """
        Forward pass with bidirectional cross-attention.
        
        Args:
            xa: Acoustic features, shape (batch, time_a, dim)
            xl: Semantic features, shape (batch, time_l, dim)
            
        Returns:
            tuple: (fused_acoustic, fused_semantic)
                - fused_acoustic: shape (batch, time_a, dim)
                - fused_semantic: shape (batch, time_l, dim)
        """
        # Acoustic attends to semantic
        q_a = self.q_proj(xa)
        k_l = self.k_proj(xl)
        v_l = self.v_proj(xl)
        ca = self.attention(q_a, k_l, v_l)

        # Semantic attends to acoustic
        q_l = self.q_proj(xl)
        k_a = self.k_proj(xa)
        v_a = self.v_proj(xa)
        cl = self.attention(q_l, k_a, v_a)

        # Apply temporal gated fusion
        fused_a = self.temporal_fusion(xa, ca)
        fused_l = self.temporal_fusion(xl, cl)

        return fused_a, fused_l

class EnhancedSERModel(nn.Module):
    """
    Enhanced Speech Emotion Recognition Model.
    
    This model combines acoustic features from HuBERT and semantic features from BERT
    using cross-modal gated interaction. The fused features are used for emotion
    classification into 4 categories (happy, sad, angry, neutral).
    
    Architecture:
        1. Extract acoustic features using HuBERT
        2. Extract semantic features using BERT (projected to match HuBERT dimension)
        3. Apply cross-modal gated interaction (CmGI)
        4. Average-pool temporal features
        5. Concatenate acoustic and semantic representations
        6. Classify emotions using fully connected layer
    
    Args:
        hubert_model: Pre-trained HuBERT model
        bert_model: Pre-trained BERT model
        num_classes (int): Number of emotion classes (default: 4)
    """
    
    def __init__(self, hubert_model, bert_model, num_classes):
        super(EnhancedSERModel, self).__init__()
        self.hubert = hubert_model
        self.bert = bert_model
        self.cmg = CmGI(dim=1024)
        self.proj_bert = nn.Linear(768, 1024)  # Project BERT to HuBERT dimension
        self.fc = nn.Linear(1024 * 2, num_classes)

    def forward(self, audio_input, text_input):
        """
        Forward pass through the model.
        
        Args:
            audio_input: Audio waveform tensor, shape (batch, time)
            text_input: Dictionary with:
                - input_ids: Token IDs, shape (batch, seq_len)
                - attention_mask: Attention mask, shape (batch, seq_len)
        
        Returns:
            Emotion class logits, shape (batch, num_classes)
        """
        # Extract features
        xa = self.hubert(audio_input).last_hidden_state  # (batch, time_a, 1024)
        xl = self.bert(**text_input).last_hidden_state   # (batch, time_l, 768)
        xl = self.proj_bert(xl)                          # (batch, time_l, 1024)
        
        # Apply cross-modal fusion
        fused_a, fused_l = self.cmg(xa, xl)
        
        # Average pooling over time
        mp_a = fused_a.mean(dim=1)  # (batch, 1024)
        mp_l = fused_l.mean(dim=1)  # (batch, 1024)
        
        # Concatenate and classify
        features = torch.cat([mp_a, mp_l], dim=-1)  # (batch, 2048)
        output = self.fc(features)                  # (batch, num_classes)
        
        return output
