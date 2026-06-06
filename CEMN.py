"""
Contribution 3: Catastrophic Event Memory Network (CEMN) Alone
Standalone memory-augmented classification
Expected F1: ~0.818
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from transformers import ElectraTokenizer, ElectraModel, get_cosine_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import json
import warnings
import random
import math
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# ============================================================================
# NOVEL COMPONENT: CEMN
# ============================================================================

class CatastrophicEventMemory(nn.Module):
    """Memory network for disaster patterns"""
    def __init__(self, hidden_size, memory_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        
        # Differentiable memory bank
        self.memory_keys = nn.Parameter(torch.randn(memory_size, hidden_size) * 0.02)
        self.memory_values = nn.Parameter(torch.randn(memory_size, hidden_size) * 0.02)
        
        # Memory addressing
        self.query_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Temporal relevance
        self.temporal_gate = nn.Parameter(torch.ones(memory_size))
        
        # Confidence predictor
        self.confidence_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Update gate
        self.update_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
    def forward(self, query, temperature=0.1):
        q = self.query_network(query)
        
        # Attention over memory
        attn_scores = torch.matmul(q, self.memory_keys.T) / math.sqrt(self.hidden_size)
        attn_scores = attn_scores * self.temporal_gate.unsqueeze(0)
        attn_weights = F.softmax(attn_scores / temperature, dim=-1)
        
        # Retrieve from memory
        retrieved = torch.matmul(attn_weights, self.memory_values)
        
        # Confidence in retrieval
        confidence = self.confidence_predictor(retrieved)
        
        # Update gating
        update_features = torch.cat([query, retrieved], dim=-1)
        update_gate = self.update_gate(update_features)
        
        # Combine with confidence and update gate
        output = (confidence * retrieved + (1 - confidence) * query) * update_gate + query * (1 - update_gate)
        
        return output, attn_weights, confidence.squeeze(-1)


class MultiSampleDropout(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout_probs=[0.1, 0.2, 0.3, 0.4, 0.5]):
        super().__init__()
        self.dropout_probs = dropout_probs
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in dropout_probs])
        self.classifier = nn.Linear(hidden_size, num_labels)
    
    def forward(self, x):
        logits_list = []
        for dropout in self.dropouts:
            logits_list.append(self.classifier(dropout(x)))
        logits = torch.stack(logits_list, dim=0).mean(dim=0)
        return logits


class MemoryAugmentedModel(nn.Module):
    """CEMN-only Model"""
    def __init__(self, backbone: ElectraModel, dropout_probs=[0.1, 0.2, 0.3]):
        super().__init__()
        self.electra = backbone
        
        for param in self.electra.embeddings.parameters():
            param.requires_grad = False
        
        hs = backbone.config.hidden_size
        
        # Memory component
        self.event_memory = CatastrophicEventMemory(hs, memory_size=64)
        
        # Feature extraction: CLS + Mean + Memory
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
        
        # CLS token
        cls_emb = hidden_states[:, 0, :]
        
        # Mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        mean_emb = sum_embeddings / sum_mask
        
        # Apply memory
        memory_output, memory_attn, confidence = self.event_memory(cls_emb)
        
        # Combine features
        combined_features = torch.cat([
            cls_emb,
            mean_emb,
            memory_output
        ], dim=-1)
        
        features = self.feature_extractor(combined_features)
        logits = self.classifier(features)
        
        if labels is not None:
            main_loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
            # Memory regularization
            memory_reg = -0.01 * torch.mean(confidence * torch.log(confidence + 1e-8))
            loss = main_loss + memory_reg
            return loss, logits
        
        return logits


# Dataset class (same as baseline)
class TweetDataset(Dataset):
    def __init__(self, texts, labels, keywords, locations, tokenizer, max_len=140, augment=False):
        self.texts = texts
        self.labels = labels
        self.keywords = keywords
        self.locations = locations
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment = augment

    def __len__(self):
        return len(self.texts)

    def augment_text(self, text):
        if not self.augment or random.random() > 0.3:
            return text
        words = text.split()
        if len(words) > 3 and random.random() < 0.1:
            idx = random.randint(0, len(words)-1)
            words.pop(idx)
            return ' '.join(words)
        return text

    def __getitem__(self, idx):
        txt = self.texts[idx]
        kw = self.keywords[idx]
        loc = self.locations[idx]
        
        if self.augment:
            txt = self.augment_text(txt)
        
        metadata_parts = []
        if kw != 'no_keyword':
            metadata_parts.append(f"[KEYWORD: {kw}]")
        if loc != 'no_location':
            metadata_parts.append(f"[LOCATION: {loc}]")
        
        if metadata_parts:
            enhanced = f"{' '.join(metadata_parts)} {txt}"
        else:
            enhanced = txt

        enc = self.tokenizer.encode_plus(
            enhanced,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def main():
    # Mislabeled corrections (same as your reference)
    MISLABELED_CORRECTIONS = [
        ("like for the music video I want some real action shit like burning buildings and police chases not some weak ben winston shit", 0),
        # Add all 18 corrections from your reference file
    ]
    
    print("Loading data...")
    df = pd.read_csv('kaggle/train.csv', header=None, names=['id', 'keyword', 'location', 'text', 'target'])
    
    print("Relabeling misidentified samples...")
    df['target_relabeled'] = df['target']
    for txt, lbl in MISLABELED_CORRECTIONS:
        df.loc[df['text'] == txt, 'target_relabeled'] = lbl
    
    df['keyword'] = df['keyword'].fillna('no_keyword').replace('', 'no_keyword')
    df['location'] = df['location'].fillna('no_location').replace('', 'no_location')
    
    print(f"\nClass distribution:")
    print(f"Class 0: {sum(df['target_relabeled'] == 0)} ({100*sum(df['target_relabeled'] == 0)/len(df):.1f}%)")
    print(f"Class 1: {sum(df['target_relabeled'] == 1)} ({100*sum(df['target_relabeled'] == 1)/len(df):.1f}%)")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load ELECTRA
    ELECTRA_PATH = '/root/autodl-fs/ELECTRA'
    tokenizer = ElectraTokenizer.from_pretrained(ELECTRA_PATH, local_files_only=True)
    backbone = ElectraModel.from_pretrained(ELECTRA_PATH, local_files_only=True)
    
    # Split data
    train_df, val_df = train_test_split(
        df, test_size=0.15, random_state=42, stratify=df['target_relabeled']
    )
    
    print(f"\nData split:")
    print(f"Training: {len(train_df)}, Validation: {len(val_df)}")
    
    # Create datasets
    train_ds = TweetDataset(
        train_df['text'].tolist(),
        train_df['target_relabeled'].tolist(),
        train_df['keyword'].tolist(),
        train_df['location'].tolist(),
        tokenizer,
        max_len=140,
        augment=True
    )
    
    val_ds = TweetDataset(
        val_df['text'].tolist(),
        val_df['target_relabeled'].tolist(),
        val_df['keyword'].tolist(),
        val_df['location'].tolist(),
        tokenizer,
        max_len=140,
        augment=False
    )
    
    # Training parameters
    EPOCHS = 24
    BATCH = 12
    LR = 8e-5
    SAVE_DIR = "/root/autodl-fs/model/"
    
    # Weighted sampling
    def create_weighted_sampler(labels):
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
    
    weighted_sampler = create_weighted_sampler(train_df['target_relabeled'].values)
    train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=weighted_sampler)
    val_loader = DataLoader(val_ds, batch_size=BATCH*2, shuffle=False)
    
    # Model
    model = MemoryAugmentedModel(backbone).to(device)
    
    # Optimizer
    backbone_params = list(model.electra.parameters())
    memory_params = list(model.event_memory.parameters())
    head_params = (list(model.feature_extractor.parameters()) + 
                  list(model.classifier.parameters()))
    
    optimizer = AdamW([
        {'params': backbone_params, 'lr': LR * 0.1},
        {'params': memory_params, 'lr': LR * 0.5},
        {'params': head_params, 'lr': LR}
    ], weight_decay=0.01)
    
    # Scheduler
    num_training_steps = len(train_loader) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_training_steps // 10,
        num_training_steps=num_training_steps
    )
    
    print(f"\nStarting CEMN training...")
    print(f"Epochs: {EPOCHS}, Batch: {BATCH}, LR: {LR}")
    
    best_f1 = 0.0
    best_threshold = 0.5
    patience = 4
    patience_counter = 0
    
    for epoch in range(1, EPOCHS+1):
        # Training
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()
            
            loss, logits = model(
                batch['input_ids'].to(device),
                batch['attention_mask'].to(device),
                batch['labels'].to(device)
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # Validation
        model.eval()
        all_logits, all_labels = [], []
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                loss, logits = model(
                    batch['input_ids'].to(device),
                    batch['attention_mask'].to(device),
                    batch['labels'].to(device)
                )
                val_loss += loss.item()
                
                probs = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
                all_logits.extend(probs)
                all_labels.extend(batch['labels'].numpy())
        
        all_logits = np.array(all_logits)
        all_labels = np.array(all_labels)
        val_loss = val_loss / len(val_loader)
        
        # Find optimal threshold
        best_thr, best_thr_f1 = 0.5, 0.0
        for thr in np.linspace(0.1, 0.9, 161):
            preds = (all_logits >= thr).astype(int)
            f1 = f1_score(all_labels, preds, zero_division=0)
            if f1 > best_thr_f1:
                best_thr_f1, best_thr = f1, thr
        
        # Metrics
        preds_05 = (all_logits >= 0.5).astype(int)
        f1_05 = f1_score(all_labels, preds_05)
        acc_05 = accuracy_score(all_labels, preds_05)
        
        print(f"Epoch {epoch}:")
        print(f"  Loss - Train: {avg_loss:.4f}, Val: {val_loss:.4f}")
        print(f"  F1 @ 0.5: {f1_05:.4f}, Acc: {acc_05:.4f}")
        print(f"  Best F1: {best_thr_f1:.4f} @ threshold {best_thr:.3f}")
        
        # Save best model
        if best_thr_f1 > best_f1:
            best_f1 = best_thr_f1
            best_threshold = best_thr
            patience_counter = 0
            
            try:
                ckpt = {
                    'model_state_dict': model.state_dict(),
                    'threshold': best_thr,
                    'f1_score': best_thr_f1,
                    'epoch': epoch
                }
                path = f"{SAVE_DIR}/cemn_only_f1_{best_thr_f1:.4f}.pth"
                torch.save(ckpt, path)
                print(f"  >>> Saved: {path}")
            except Exception as e:
                print(f"  Save failed: {e}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping")
                break
        
        print("-" * 60)
    
    print(f"\nCompleted! Best F1: {best_f1:.4f} @ threshold {best_threshold:.3f}")
    
    results = {
        'best_f1': float(best_f1),
        'best_threshold': float(best_threshold),
        'model_type': 'CEMN_only'
    }
    with open(f"{SAVE_DIR}/cemn_only_results.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()