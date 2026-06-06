import argparse
import glob
import os
import warnings
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import ElectraModel, ElectraTokenizer


warnings.filterwarnings("ignore")


HIERARCHICAL_NAMES = ["no_keyword", "disaster_keyword", "other_keyword"]


class TestTweetDataset(Dataset):
    def __init__(self, texts, keywords, locations, tokenizer, max_len=140, multi_channel=False):
        self.texts = texts
        self.keywords = keywords
        self.locations = locations
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.multi_channel = multi_channel

    def __len__(self):
        return len(self.texts)

    def _enhanced_text(self, text, keyword, location):
        metadata_parts = []
        if keyword != "no_keyword":
            metadata_parts.append(f"[KEYWORD: {keyword}]")
        if location != "no_location":
            metadata_parts.append(f"[LOCATION: {location}]")
        return f"{' '.join(metadata_parts)} {text}" if metadata_parts else text

    def _encode(self, text, max_length):
        return self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

    def __getitem__(self, idx):
        text = self.texts[idx]
        keyword = self.keywords[idx]
        location = self.locations[idx]

        enc = self._encode(self._enhanced_text(text, keyword, location), self.max_len)
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }

        if self.multi_channel:
            kw_enc = self._encode(keyword if keyword != "no_keyword" else "", 32)
            loc_enc = self._encode(location if location != "no_location" else "", 64)
            item.update(
                {
                    "kw_input_ids": kw_enc["input_ids"].squeeze(0),
                    "kw_attention_mask": kw_enc["attention_mask"].squeeze(0),
                    "loc_input_ids": loc_enc["input_ids"].squeeze(0),
                    "loc_attention_mask": loc_enc["attention_mask"].squeeze(0),
                }
            )

        return item


class MultiSampleDropout(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout_probs=[0.1, 0.2, 0.3, 0.4, 0.5]):
        super().__init__()
        self.dropout_probs = dropout_probs
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in dropout_probs])
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        logits_list = [self.classifier(dropout(x)) for dropout in self.dropouts]
        return torch.stack(logits_list, dim=0).mean(dim=0)


class SimplifiedElectraModel(nn.Module):
    def __init__(self, backbone: ElectraModel, dropout_probs=[0.1, 0.2, 0.3]):
        super().__init__()
        self.electra = backbone
        for param in self.electra.embeddings.parameters():
            param.requires_grad = False

        hs = backbone.config.hidden_size
        self.feature_extractor = nn.Sequential(
            nn.Linear(hs * 2, hs),
            nn.GELU(),
            nn.LayerNorm(hs),
            nn.Dropout(0.1),
            nn.Linear(hs, hs // 2),
            nn.GELU(),
            nn.LayerNorm(hs // 2),
        )
        self.classifier = MultiSampleDropout(hs // 2, 2, dropout_probs)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        cls_emb = hidden_states[:, 0, :]

        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        mean_emb = sum_embeddings / sum_mask

        features = self.feature_extractor(torch.cat([cls_emb, mean_emb], dim=-1))
        logits = self.classifier(features)
        if labels is not None:
            loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
            return loss, logits
        return logits


class MultiChannelElectraModel(nn.Module):
    def __init__(self, backbone: ElectraModel, dropout_probs=[0.1, 0.2, 0.3]):
        super().__init__()
        self.electra = backbone
        hs = backbone.config.hidden_size

        self.kw_encoder = nn.Sequential(
            nn.Linear(hs, hs // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hs // 2, hs),
        )
        self.loc_encoder = nn.Sequential(
            nn.Linear(hs, hs // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hs // 2, hs),
        )

        for param in self.electra.embeddings.parameters():
            param.requires_grad = False

        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=hs,
            num_heads=8,
            batch_first=True,
            dropout=0.1,
        )
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=hs,
            num_heads=4,
            batch_first=True,
            dropout=0.1,
        )
        self.fusion_gate = nn.Sequential(nn.Linear(hs * 3, hs), nn.Sigmoid())
        self.feature_extractor = nn.Sequential(
            nn.Linear(hs * 2, hs * 2),
            nn.GELU(),
            nn.LayerNorm(hs * 2),
            nn.Dropout(0.1),
            nn.Linear(hs * 2, hs),
            nn.GELU(),
            nn.LayerNorm(hs),
        )
        self.classifier = MultiSampleDropout(hs, 2, dropout_probs)

    def forward(
        self,
        input_ids,
        attention_mask,
        kw_input_ids,
        kw_attention_mask,
        loc_input_ids,
        loc_attention_mask,
        labels=None,
    ):
        outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        cls_emb = hidden_states[:, 0, :]

        attn_emb, _ = self.attention_pooling(
            hidden_states,
            hidden_states,
            hidden_states,
            key_padding_mask=~attention_mask.bool(),
        )
        attn_emb = attn_emb.mean(dim=1)

        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        mean_emb = sum_embeddings / sum_mask
        main_features = torch.cat([cls_emb, attn_emb, mean_emb], dim=-1)

        kw_outputs = self.electra(input_ids=kw_input_ids, attention_mask=kw_attention_mask)
        loc_outputs = self.electra(input_ids=loc_input_ids, attention_mask=loc_attention_mask)
        kw_features = self.kw_encoder(kw_outputs.last_hidden_state[:, 0, :])
        loc_features = self.loc_encoder(loc_outputs.last_hidden_state[:, 0, :])

        aux_features = torch.stack([kw_features, loc_features], dim=1)
        cross_attended, _ = self.cross_modal_attention(
            cls_emb.unsqueeze(1), aux_features, aux_features
        )
        cross_attended = cross_attended.squeeze(1)

        gate = self.fusion_gate(main_features)
        fused_features = torch.cat(
            [
                main_features.mean(dim=-1, keepdim=True).expand(-1, cls_emb.size(-1)),
                cross_attended * gate + cls_emb * (1 - gate),
            ],
            dim=-1,
        )
        features = self.feature_extractor(fused_features)
        logits = self.classifier(features)
        if labels is not None:
            loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
            return loss, logits
        return logits


class EnhancedHierarchicalElectraModel(nn.Module):
    def __init__(self, backbone: ElectraModel, dropout_probs=[0.1, 0.2, 0.3]):
        super().__init__()
        self.electra = backbone
        for param in self.electra.embeddings.parameters():
            param.requires_grad = False

        hs = backbone.config.hidden_size
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=hs,
            num_heads=8,
            batch_first=True,
            dropout=0.1,
        )
        self.shared_features = nn.Sequential(
            nn.Linear(hs * 3, hs * 2),
            nn.GELU(),
            nn.LayerNorm(hs * 2),
            nn.Dropout(0.1),
        )
        self.main_head = nn.Sequential(
            nn.Linear(hs * 2, hs),
            nn.GELU(),
            nn.LayerNorm(hs),
            MultiSampleDropout(hs, 2, dropout_probs),
        )
        self.hier_head = nn.Sequential(
            nn.Linear(hs * 2, hs // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hs // 2, 3),
        )

    def forward(self, input_ids, attention_mask, labels=None, hier_labels=None):
        outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        cls_emb = hidden_states[:, 0, :]

        attn_emb, _ = self.attention_pooling(
            hidden_states,
            hidden_states,
            hidden_states,
            key_padding_mask=~attention_mask.bool(),
        )
        attn_emb = attn_emb.mean(dim=1)

        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        mean_emb = sum_embeddings / sum_mask

        shared_feats = self.shared_features(torch.cat([cls_emb, attn_emb, mean_emb], dim=-1))
        main_logits = self.main_head(shared_feats)
        hier_logits = self.hier_head(shared_feats)

        if labels is not None:
            main_loss = F.cross_entropy(main_logits, labels, label_smoothing=0.1)
            if hier_labels is not None:
                hier_loss = F.cross_entropy(hier_logits, hier_labels)
                return main_loss + 0.15 * hier_loss, main_logits, hier_logits
            return main_loss, main_logits, hier_logits
        return main_logits, hier_logits


class CombinedElectraModel(nn.Module):
    def __init__(self, backbone: ElectraModel, dropout_probs=[0.1, 0.2, 0.3]):
        super().__init__()
        self.electra = backbone
        hs = backbone.config.hidden_size

        self.kw_encoder = nn.Sequential(
            nn.Linear(hs, hs // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hs // 2, hs),
        )
        self.loc_encoder = nn.Sequential(
            nn.Linear(hs, hs // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hs // 2, hs),
        )

        for param in self.electra.embeddings.parameters():
            param.requires_grad = False

        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=hs,
            num_heads=8,
            batch_first=True,
            dropout=0.1,
        )
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=hs,
            num_heads=4,
            batch_first=True,
            dropout=0.1,
        )
        self.fusion_gate = nn.Sequential(nn.Linear(hs * 3, hs), nn.Sigmoid())
        self.shared_features = nn.Sequential(
            nn.Linear(hs * 2, hs * 2),
            nn.GELU(),
            nn.LayerNorm(hs * 2),
            nn.Dropout(0.1),
            nn.Linear(hs * 2, hs),
            nn.GELU(),
            nn.LayerNorm(hs),
        )
        self.main_head = MultiSampleDropout(hs, 2, dropout_probs)
        self.hier_head = nn.Sequential(
            nn.Linear(hs, hs // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hs // 2, 3),
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        kw_input_ids,
        kw_attention_mask,
        loc_input_ids,
        loc_attention_mask,
        labels=None,
        hier_labels=None,
    ):
        outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        cls_emb = hidden_states[:, 0, :]

        attn_emb, _ = self.attention_pooling(
            hidden_states,
            hidden_states,
            hidden_states,
            key_padding_mask=~attention_mask.bool(),
        )
        attn_emb = attn_emb.mean(dim=1)

        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        mean_emb = sum_embeddings / sum_mask
        main_features = torch.cat([cls_emb, attn_emb, mean_emb], dim=-1)

        kw_outputs = self.electra(input_ids=kw_input_ids, attention_mask=kw_attention_mask)
        loc_outputs = self.electra(input_ids=loc_input_ids, attention_mask=loc_attention_mask)
        kw_features = self.kw_encoder(kw_outputs.last_hidden_state[:, 0, :])
        loc_features = self.loc_encoder(loc_outputs.last_hidden_state[:, 0, :])

        aux_features = torch.stack([kw_features, loc_features], dim=1)
        cross_attended, _ = self.cross_modal_attention(
            cls_emb.unsqueeze(1), aux_features, aux_features
        )
        cross_attended = cross_attended.squeeze(1)

        gate = self.fusion_gate(main_features)
        fused_features = torch.cat(
            [
                main_features.mean(dim=-1, keepdim=True).expand(-1, cls_emb.size(-1)),
                cross_attended * gate + cls_emb * (1 - gate),
            ],
            dim=-1,
        )
        shared_feats = self.shared_features(fused_features)
        main_logits = self.main_head(shared_feats)
        hier_logits = self.hier_head(shared_feats)

        if labels is not None:
            main_loss = F.cross_entropy(main_logits, labels, label_smoothing=0.1)
            if hier_labels is not None:
                hier_loss = F.cross_entropy(hier_logits, hier_labels)
                return main_loss + 0.15 * hier_loss, main_logits, hier_logits
            return main_loss, main_logits, hier_logits
        return main_logits, hier_logits


class CatastrophicEventMemory(nn.Module):
    def __init__(self, hidden_size, memory_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.memory_keys = nn.Parameter(torch.randn(memory_size, hidden_size) * 0.02)
        self.memory_values = nn.Parameter(torch.randn(memory_size, hidden_size) * 0.02)
        self.query_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
        )
        self.temporal_gate = nn.Parameter(torch.ones(memory_size))
        self.confidence_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )
        self.update_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, query, temperature=0.1):
        q = self.query_network(query)
        attn_scores = torch.matmul(q, self.memory_keys.T) / math.sqrt(self.hidden_size)
        attn_scores = attn_scores * self.temporal_gate.unsqueeze(0)
        attn_weights = F.softmax(attn_scores / temperature, dim=-1)
        retrieved = torch.matmul(attn_weights, self.memory_values)
        confidence = self.confidence_predictor(retrieved)
        update_features = torch.cat([query, retrieved], dim=-1)
        update_gate = self.update_gate(update_features)
        output = (
            (confidence * retrieved + (1 - confidence) * query) * update_gate
            + query * (1 - update_gate)
        )
        return output, attn_weights, confidence.squeeze(-1)


class MemoryAugmentedModel(nn.Module):
    def __init__(self, backbone: ElectraModel, dropout_probs=[0.1, 0.2, 0.3]):
        super().__init__()
        self.electra = backbone
        for param in self.electra.embeddings.parameters():
            param.requires_grad = False

        hs = backbone.config.hidden_size
        self.event_memory = CatastrophicEventMemory(hs, memory_size=64)
        self.feature_extractor = nn.Sequential(
            nn.Linear(hs * 3, hs * 2),
            nn.GELU(),
            nn.LayerNorm(hs * 2),
            nn.Dropout(0.1),
            nn.Linear(hs * 2, hs),
            nn.GELU(),
            nn.LayerNorm(hs),
            nn.Dropout(0.1),
            nn.Linear(hs, hs // 2),
            nn.GELU(),
            nn.LayerNorm(hs // 2),
        )
        self.classifier = MultiSampleDropout(hs // 2, 2, dropout_probs)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        cls_emb = hidden_states[:, 0, :]

        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        mean_emb = sum_embeddings / sum_mask

        memory_output, _, confidence = self.event_memory(cls_emb)
        features = self.feature_extractor(torch.cat([cls_emb, mean_emb, memory_output], dim=-1))
        logits = self.classifier(features)
        if labels is not None:
            main_loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
            memory_reg = -0.01 * torch.mean(confidence * torch.log(confidence + 1e-8))
            return main_loss + memory_reg, logits
        return logits


class AuxMC_CEMN_Model(nn.Module):
    def __init__(self, backbone: ElectraModel, dropout_probs=[0.1, 0.2, 0.3]):
        super().__init__()
        self.electra = backbone
        for param in self.electra.embeddings.parameters():
            param.requires_grad = False

        hs = backbone.config.hidden_size
        self.kw_encoder = nn.Sequential(
            nn.Linear(hs, hs // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hs // 2, hs),
        )
        self.loc_encoder = nn.Sequential(
            nn.Linear(hs, hs // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hs // 2, hs),
        )
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=hs,
            num_heads=8,
            batch_first=True,
            dropout=0.1,
        )
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=hs,
            num_heads=4,
            batch_first=True,
            dropout=0.1,
        )
        self.fusion_gate = nn.Sequential(nn.Linear(hs * 3, hs), nn.Sigmoid())
        self.hier_head = nn.Sequential(
            nn.Linear(hs, hs // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hs // 2, 3),
        )
        self.event_memory = CatastrophicEventMemory(hs, memory_size=32)
        self.shared_features = nn.Sequential(
            nn.Linear(hs * 4, hs * 2),
            nn.GELU(),
            nn.LayerNorm(hs * 2),
            nn.Dropout(0.1),
            nn.Linear(hs * 2, hs),
            nn.GELU(),
            nn.LayerNorm(hs),
            nn.Dropout(0.1),
            nn.Linear(hs, hs),
            nn.GELU(),
            nn.LayerNorm(hs),
        )
        self.main_head = MultiSampleDropout(hs, 2, dropout_probs)

    def forward(
        self,
        input_ids,
        attention_mask,
        kw_input_ids,
        kw_attention_mask,
        loc_input_ids,
        loc_attention_mask,
        labels=None,
        hier_labels=None,
    ):
        outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        cls_emb = hidden_states[:, 0, :]

        attn_emb, _ = self.attention_pooling(
            hidden_states,
            hidden_states,
            hidden_states,
            key_padding_mask=~attention_mask.bool(),
        )
        attn_emb = attn_emb.mean(dim=1)

        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        mean_emb = sum_embeddings / sum_mask

        main_features = torch.cat([cls_emb, attn_emb, mean_emb], dim=-1)
        kw_outputs = self.electra(input_ids=kw_input_ids, attention_mask=kw_attention_mask)
        loc_outputs = self.electra(input_ids=loc_input_ids, attention_mask=loc_attention_mask)
        kw_features = self.kw_encoder(kw_outputs.last_hidden_state[:, 0, :])
        loc_features = self.loc_encoder(loc_outputs.last_hidden_state[:, 0, :])

        aux_features = torch.stack([kw_features, loc_features], dim=1)
        cross_attended, _ = self.cross_modal_attention(
            cls_emb.unsqueeze(1), aux_features, aux_features
        )
        cross_attended = cross_attended.squeeze(1)
        gate = self.fusion_gate(main_features)
        fused_mc = cross_attended * gate + cls_emb * (1 - gate)
        memory_output, _, confidence = self.event_memory(fused_mc)

        combined_features = torch.cat(
            [
                main_features.mean(dim=-1, keepdim=True).expand(-1, cls_emb.size(-1)),
                fused_mc,
                memory_output,
            ],
            dim=-1,
        )
        shared_feats = self.shared_features(combined_features)
        main_logits = self.main_head(shared_feats)
        hier_logits = self.hier_head(shared_feats)

        if labels is not None:
            main_loss = F.cross_entropy(main_logits, labels, label_smoothing=0.1)
            if hier_labels is not None:
                hier_loss = F.cross_entropy(hier_logits, hier_labels)
                total_loss = main_loss + 0.15 * hier_loss
            else:
                total_loss = main_loss
            memory_reg = -0.01 * torch.mean(confidence * torch.log(confidence + 1e-8))
            return total_loss + memory_reg, main_logits, hier_logits
        return main_logits, hier_logits


MODEL_CONFIGS = {
    "baseline": {
        "name": "Baseline simplified ELECTRA",
        "model_class": SimplifiedElectraModel,
        "checkpoint_pattern": "simplified_model_f1_*.pth",
        "multi_channel": False,
        "hierarchical": False,
    },
    "mc": {
        "name": "Multi-channel ELECTRA",
        "model_class": MultiChannelElectraModel,
        "checkpoint_pattern": "multichannel_only_f1_*.pth",
        "multi_channel": True,
        "hierarchical": False,
    },
    "aux": {
        "name": "Auxiliary hierarchical ELECTRA",
        "model_class": EnhancedHierarchicalElectraModel,
        "checkpoint_pattern": "enhanced_hierarchical_f1_*.pth",
        "multi_channel": False,
        "hierarchical": True,
    },
    "aux_mc": {
        "name": "Auxiliary + multi-channel ELECTRA",
        "model_class": CombinedElectraModel,
        "checkpoint_pattern": "combined_multichannel_hierarchical_f1_*.pth",
        "multi_channel": True,
        "hierarchical": True,
    },
    "cemn": {
        "name": "CEMN-only ELECTRA",
        "model_class": MemoryAugmentedModel,
        "checkpoint_pattern": "cemn_only_f1_*.pth",
        "multi_channel": False,
        "hierarchical": False,
    },
    "aux_mc_cemn": {
        "name": "Auxiliary + multi-channel + CEMN ELECTRA",
        "model_class": AuxMC_CEMN_Model,
        "checkpoint_pattern": "auxmc_cemn_f1_*.pth",
        "multi_channel": True,
        "hierarchical": True,
    },
}


def load_test_data(test_file):
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")

    try:
        test_df = pd.read_csv(test_file)
        if "text" not in test_df.columns:
            raise ValueError("No text column found")
        print("Loaded test.csv with headers")
    except Exception:
        test_df = pd.read_csv(test_file, header=None)
        if len(test_df.columns) != 4:
            raise ValueError(
                f"Unexpected test.csv column count: {len(test_df.columns)}. "
                "Expected id, keyword, location, text."
            )
        test_df.columns = ["id", "keyword", "location", "text"]
        print("Loaded test.csv without headers")

    if "id" not in test_df.columns:
        test_df["id"] = np.arange(len(test_df))
    if "keyword" not in test_df.columns:
        test_df["keyword"] = "no_keyword"
    if "location" not in test_df.columns:
        test_df["location"] = "no_location"

    test_df["text"] = test_df["text"].fillna("").astype(str)
    test_df["keyword"] = (
        test_df["keyword"].fillna("no_keyword").astype(str).replace("", "no_keyword")
    )
    test_df["location"] = (
        test_df["location"].fillna("no_location").astype(str).replace("", "no_location")
    )
    return test_df


def find_latest_checkpoint(model_dir, pattern):
    search_path = os.path.join(model_dir, pattern)
    matches = glob.glob(search_path)
    if not matches:
        raise FileNotFoundError(f"No checkpoint found for pattern: {search_path}")
    return max(matches, key=os.path.getmtime)


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model(model_key, backbone, checkpoint_path, device):
    config = MODEL_CONFIGS[model_key]
    model = config["model_class"](backbone).to(device)
    checkpoint = load_checkpoint(checkpoint_path, device)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)

    threshold = checkpoint.get("threshold", 0.5) if isinstance(checkpoint, dict) else 0.5
    f1_score = checkpoint.get("f1_score") if isinstance(checkpoint, dict) else None
    epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None

    print(f"Loaded checkpoint: {checkpoint_path}")
    if epoch is not None:
        print(f"Checkpoint epoch: {epoch}")
    if f1_score is not None:
        print(f"Validation F1: {f1_score:.4f}")
    print(f"Saved threshold: {threshold:.4f}")
    return model, threshold


def predict(model, loader, device, multi_channel=False, hierarchical=False, threshold=0.5):
    model.eval()
    predictions = []
    probabilities = []
    hierarchical_predictions = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            if multi_channel:
                outputs = model(
                    input_ids,
                    attention_mask,
                    batch["kw_input_ids"].to(device),
                    batch["kw_attention_mask"].to(device),
                    batch["loc_input_ids"].to(device),
                    batch["loc_attention_mask"].to(device),
                )
            else:
                outputs = model(input_ids, attention_mask)

            if isinstance(outputs, tuple):
                main_logits = outputs[0]
                hier_logits = outputs[1] if hierarchical and len(outputs) > 1 else None
            else:
                main_logits = outputs
                hier_logits = None

            probs = torch.softmax(main_logits, dim=1)[:, 1].cpu().numpy()
            preds = (probs >= threshold).astype(int)
            predictions.extend(preds.tolist())
            probabilities.extend(probs.tolist())

            if hier_logits is not None:
                hierarchical_predictions.extend(
                    torch.argmax(hier_logits, dim=1).cpu().numpy().tolist()
                )

    return predictions, probabilities, hierarchical_predictions


def write_outputs(
    test_df,
    predictions,
    probabilities,
    hierarchical_predictions,
    output_dir,
    submission_name,
    details_name,
):
    os.makedirs(output_dir, exist_ok=True)

    submission = pd.DataFrame({"id": test_df["id"], "target": predictions})
    submission_path = os.path.join(output_dir, submission_name)
    submission.to_csv(submission_path, index=False)

    detailed = pd.DataFrame(
        {
            "id": test_df["id"],
            "text": test_df["text"],
            "keyword": test_df["keyword"],
            "location": test_df["location"],
            "probability": probabilities,
            "prediction": predictions,
        }
    )
    if hierarchical_predictions:
        detailed["hierarchical_class"] = [
            HIERARCHICAL_NAMES[idx] for idx in hierarchical_predictions
        ]

    details_path = os.path.join(output_dir, details_name)
    detailed.to_csv(details_path, index=False)
    return submission_path, details_path, detailed


def print_prediction_summary(predictions, probabilities, hierarchical_predictions, threshold, detailed):
    total = len(predictions)
    disaster_count = int(sum(predictions))
    non_disaster_count = total - disaster_count

    print("\n" + "=" * 50)
    print("PREDICTION STATISTICS")
    print("=" * 50)
    print(f"Total samples: {total}")
    print(f"Predicted disaster (1): {disaster_count} ({100 * disaster_count / total:.1f}%)")
    print(
        f"Predicted not disaster (0): "
        f"{non_disaster_count} ({100 * non_disaster_count / total:.1f}%)"
    )
    print(f"Probability mean: {np.mean(probabilities):.4f}")
    print(f"Probability std: {np.std(probabilities):.4f}")
    print(f"Probability min: {np.min(probabilities):.4f}")
    print(f"Probability max: {np.max(probabilities):.4f}")
    print(f"Threshold used: {threshold:.4f}")

    if hierarchical_predictions:
        print("\n" + "=" * 50)
        print("HIERARCHICAL PREDICTION STATISTICS")
        print("=" * 50)
        for idx, name in enumerate(HIERARCHICAL_NAMES):
            count = sum(1 for pred in hierarchical_predictions if pred == idx)
            print(f"{name}: {count} ({100 * count / len(hierarchical_predictions):.1f}%)")

    print("\n" + "=" * 50)
    print("SAMPLE PREDICTIONS")
    print("=" * 50)
    for idx in range(min(10, len(detailed))):
        row = detailed.iloc[idx]
        label = "DISASTER" if row["prediction"] == 1 else "NOT DISASTER"
        text = row["text"][:100] + ("..." if len(row["text"]) > 100 else "")
        print(f"\n[{idx + 1}] ID: {row['id']}")
        print(f"Text: {text}")
        print(f"Keyword: {row['keyword']}")
        print(f"Prediction: {label} (prob: {row['probability']:.4f})")
        if "hierarchical_class" in row:
            print(f"Hierarchical: {row['hierarchical_class']}")


def run_inference(args):
    config = MODEL_CONFIGS[args.model_key]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model: {config['name']}")
    print(f"Using device: {device}")

    model_path = args.model_path
    if model_path is None:
        model_path = find_latest_checkpoint(args.model_dir, config["checkpoint_pattern"])
        print(f"Auto-selected latest checkpoint: {model_path}")
    elif not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not os.path.exists(args.electra_path):
        raise FileNotFoundError(f"ELECTRA model not found: {args.electra_path}")

    test_df = load_test_data(args.test_file)
    print(f"Test dataset shape: {test_df.shape}")

    print("\nLoading ELECTRA tokenizer and backbone...")
    tokenizer = ElectraTokenizer.from_pretrained(args.electra_path, local_files_only=True)
    backbone = ElectraModel.from_pretrained(args.electra_path, local_files_only=True)

    print("\nLoading trained model...")
    model, saved_threshold = load_model(args.model_key, backbone, model_path, device)
    threshold = args.threshold if args.threshold is not None else saved_threshold
    print(f"Using threshold: {threshold:.4f}")

    dataset = TestTweetDataset(
        test_df["text"].tolist(),
        test_df["keyword"].tolist(),
        test_df["location"].tolist(),
        tokenizer,
        max_len=args.max_len,
        multi_channel=config["multi_channel"],
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print("\nMaking predictions...")
    predictions, probabilities, hierarchical_predictions = predict(
        model,
        loader,
        device,
        multi_channel=config["multi_channel"],
        hierarchical=config["hierarchical"],
        threshold=threshold,
    )

    submission_path, details_path, detailed = write_outputs(
        test_df,
        predictions,
        probabilities,
        hierarchical_predictions,
        args.output_dir,
        args.submission_name,
        args.details_name,
    )

    print(f"\nSubmission saved to: {submission_path}")
    print(f"Detailed predictions saved to: {details_path}")
    print_prediction_summary(
        predictions,
        probabilities,
        hierarchical_predictions,
        threshold,
        detailed,
    )
    print("\nTesting completed successfully.")


def run_cli(model_key, default_model_path=None):
    parser = argparse.ArgumentParser(description=f"Run Kaggle inference for {model_key}.")
    parser.add_argument("--model-key", default=model_key, choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--model-path", default=default_model_path)
    parser.add_argument("--model-dir", default="/root/autodl-fs/model")
    parser.add_argument("--test-file", default="kaggle/test.csv")
    parser.add_argument("--electra-path", default="/root/autodl-fs/ELECTRA")
    parser.add_argument("--output-dir", default="kaggle")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=140)
    parser.add_argument("--submission-name", default=f"submission_{model_key}.csv")
    parser.add_argument("--details-name", default=f"detailed_predictions_{model_key}.csv")
    run_inference(parser.parse_args())
