"""
Aux + M-C + CEMN Combined Model
Combines your best model (Aux + M-C) with Memory Network (CEMN)
Expected F1: ~0.839
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
    def __init__(self, hidden_size, memory_size=32):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.memory_keys = nn.Parameter(torch.randn(memory_size, hidden_size) * 0.02)
        self.memory_values = nn.Parameter(torch.randn(memory_size, hidden_size) * 0.02)
        self.query_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(hidden_size, hidden_size)
        )
        self.temporal_gate = nn.Parameter(torch.ones(memory_size))
        self.confidence_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(), nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(), nn.Linear(hidden_size, 1),
            nn.Sigmoid()
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
        return torch.stack(logits_list, dim=0).mean(dim=0)


class AuxMC_CEMN_Model(nn.Module):
    """Aux + M-C + CEMN Combined"""
    def __init__(self, backbone: ElectraModel, dropout_probs=[0.1, 0.2, 0.3]):
        super().__init__()
        self.electra = backbone
        
        for param in self.electra.embeddings.parameters():
            param.requires_grad = False
        
        hs = backbone.config.hidden_size
        
        # Multi-Channel components (from Aux + M-C)
        self.kw_encoder = nn.Sequential(
            nn.Linear(hs, hs // 2),
            nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hs // 2, hs)
        )
        self.loc_encoder = nn.Sequential(
            nn.Linear(hs, hs // 2),
            nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hs // 2, hs)
        )
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=hs, num_heads=8,
            batch_first=True, dropout=0.1
        )
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=hs, num_heads=4,
            batch_first=True, dropout=0.1
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(hs * 3, hs),
            nn.Sigmoid()
        )
        
        # Hierarchical auxiliary head
        self.hier_head = nn.Sequential(
            nn.Linear(hs, hs // 2),
            nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hs // 2, 3)
        )
        
        # Novel component: CEMN
        self.event_memory = CatastrophicEventMemory(hs, memory_size=32)
        
        # Shared feature extraction
        # Input: Main features (3×hs from M-C) + Memory = 4×hs
        self.shared_features = nn.Sequential(
            nn.Linear(hs * 4, hs * 2),
            nn.GELU(), nn.LayerNorm(hs * 2),
            nn.Dropout(0.1),
            nn.Linear(hs * 2, hs),
            nn.GELU(), nn.LayerNorm(hs),
            nn.Dropout(0.1),
            nn.Linear(hs, hs),
            nn.GELU(), nn.LayerNorm(hs)
        )
        
        # Main classifier
        self.main_head = MultiSampleDropout(hs, 2, dropout_probs)

    def forward(self, input_ids, attention_mask, kw_input_ids, kw_attention_mask,
                loc_input_ids, loc_attention_mask, labels=None, hier_labels=None):
        
        # Main encoding
        outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        cls_emb = hidden_states[:, 0, :]
        
        # Attention pooling
        attn_emb, _ = self.attention_pooling(
            hidden_states, hidden_states, hidden_states,
            key_padding_mask=~attention_mask.bool()
        )
        attn_emb = attn_emb.mean(dim=1)
        
        # Mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        mean_emb = sum_embeddings / sum_mask
        
        # Multi-Channel processing
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
        
        # Apply CEMN to fused multi-channel features
        memory_output, memory_attn, confidence = self.event_memory(fused_mc)
        
        # Combine all features
        combined_features = torch.cat([
            main_features.mean(dim=-1, keepdim=True).expand(-1, cls_emb.size(-1)),
            fused_mc,
            memory_output
        ], dim=-1)
        
        # Shared features and classification
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
            
            # Memory regularization
            memory_reg = -0.01 * torch.mean(confidence * torch.log(confidence + 1e-8))
            total_loss = total_loss + memory_reg
            
            return total_loss, main_logits, hier_logits
        
        return main_logits, hier_logits


# Combined Dataset (same as Aux + M-C)
class CombinedTweetDataset(Dataset):
    def __init__(self, texts, labels, keywords, locations, tokenizer, max_len=140, augment=False):
        self.texts = texts
        self.labels = labels
        self.keywords = keywords
        self.locations = locations
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment = augment
        self.hier_labels = self._create_hierarchical_labels()

    def _create_hierarchical_labels(self):
        disaster_keywords = {'earthquake', 'fire', 'flood', 'hurricane', 'tornado',
                           'explosion', 'crash', 'bomb', 'accident', 'emergency'}
        hier_labels = []
        for kw in self.keywords:
            if kw == 'no_keyword':
                hier_labels.append(0)
            elif any(dk in kw.lower() for dk in disaster_keywords):
                hier_labels.append(1)
            else:
                hier_labels.append(2)
        return hier_labels

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
        
        enhanced = f"{' '.join(metadata_parts)} {txt}" if metadata_parts else txt
        
        enc = self.tokenizer.encode_plus(
            enhanced, add_special_tokens=True,
            max_length=self.max_len, padding='max_length',
            truncation=True, return_attention_mask=True,
            return_tensors='pt'
        )
        
        kw_enc = self.tokenizer.encode_plus(
            kw if kw != 'no_keyword' else '',
            add_special_tokens=True, max_length=32,
            padding='max_length', truncation=True,
            return_attention_mask=True, return_tensors='pt'
        )
        
        loc_enc = self.tokenizer.encode_plus(
            loc if loc != 'no_location' else '',
            add_special_tokens=True, max_length=64,
            padding='max_length', truncation=True,
            return_attention_mask=True, return_tensors='pt'
        )
        
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'kw_input_ids': kw_enc['input_ids'].squeeze(0),
            'kw_attention_mask': kw_enc['attention_mask'].squeeze(0),
            'loc_input_ids': loc_enc['input_ids'].squeeze(0),
            'loc_attention_mask': loc_enc['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
            'hier_label': torch.tensor(self.hier_labels[idx], dtype=torch.long)
        }

def main():
    MISLABELED_CORRECTIONS = [
        ("like for the music video I want some real action shit like burning buildings and police chases not some weak ben winston shit", 0),
        ("Hellfire is surrounded by desires so be careful and donÛªt let your desires control you! #Afterlife",  0),
        ("To fight bioterrorism sir",  0),
        (".POTUS #StrategicPatience is a strategy for #Genocide; refugees; IDP Internally displaced people; horror; etc. https://t.co/rqWuoy1fm4",  1),
        ("CLEARED:incident with injury:I-495  inner loop Exit 31 - MD 97/Georgia Ave Silver Spring",  1),
        ("#foodscare #offers2go #NestleIndia slips into loss after #Magginoodle #ban unsafe and hazardous for #humanconsumption",  0),
        ("In #islam saving a person is equal in reward to saving all humans! Islam is the opposite of terrorism!",  0),
        ("Who is bringing the tornadoes and floods. Who is bringing the climate change. God is after America He is plaguing her\n \n#FARRAKHAN #QUOTE",  1),
        ("RT NotExplained: The only known image of infamous hijacker D.B. Cooper. http://t.co/JlzK2HdeTG",  1),
        ("Mmmmmm I'm burning.... I'm burning buildings I'm building.... Oooooohhhh oooh ooh...", 0),
        ("wowo--=== 12000 Nigerian refugees repatriated from Cameroon",  0),
        ("He came to a land which was engulfed in tribal war and turned it into a land of peace i.e. Madinah. #ProphetMuhammad #islam", 0),
        ("Hellfire! We donÛªt even want to think about it or mention it so letÛªs not do anything that leads to it #islam!", 0),
        ("The Prophet (peace be upon him) said 'Save yourself from Hellfire even if it is by giving half a date in charity.'", 0),
        ("Caution: breathing may be hazardous to your health.",  1),
        ("I Pledge Allegiance To The P.O.P.E. And The Burning Buildings of Epic City. ??????", 0),
        ("#Allah describes piling up #wealth thinking it would last #forever as the description of the people of #Hellfire in Surah Humaza. #Reflect",  0),
        ("that horrible sinking feeling when youÛªve been at home on your phone for a while and you realise its been on 3G this whole time",  0)
    ]

    
    print("Loading data...")
    df = pd.read_csv('kaggle/train.csv', header=None, names=['id', 'keyword', 'location', 'text', 'target'])
    
    # Apply corrections
    df['target_relabeled'] = df['target']
    # for txt, lbl in MISLABELED_CORRECTIONS:
    #     df.loc[df['text'] == txt, 'target_relabeled'] = lbl
    
    df['keyword'] = df['keyword'].fillna('no_keyword').replace('', 'no_keyword')
    df['location'] = df['location'].fillna('no_location').replace('', 'no_location')
    
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
    
    print(f"\nData split: Training: {len(train_df)}, Validation: {len(val_df)}")
    
    # Create datasets
    train_ds = CombinedTweetDataset(
        train_df['text'].tolist(),
        train_df['target_relabeled'].tolist(),
        train_df['keyword'].tolist(),
        train_df['location'].tolist(),
        tokenizer, max_len=140, augment=True
    )
    
    val_ds = CombinedTweetDataset(
        val_df['text'].tolist(),
        val_df['target_relabeled'].tolist(),
        val_df['keyword'].tolist(),
        val_df['location'].tolist(),
        tokenizer, max_len=140, augment=False
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
        return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    weighted_sampler = create_weighted_sampler(train_df['target_relabeled'].values)
    train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=weighted_sampler)
    val_loader = DataLoader(val_ds, batch_size=BATCH*2, shuffle=False)
    
    # Model
    model = AuxMC_CEMN_Model(backbone).to(device)
    
    # Optimizer groups
    backbone_params = list(model.electra.parameters())
    mc_aux_params = (list(model.kw_encoder.parameters()) + 
                    list(model.loc_encoder.parameters()) +
                    list(model.attention_pooling.parameters()) +
                    list(model.cross_modal_attention.parameters()) +
                    list(model.fusion_gate.parameters()) +
                    list(model.hier_head.parameters()))
    memory_params = list(model.event_memory.parameters())
    head_params = (list(model.shared_features.parameters()) +
                  list(model.main_head.parameters()))
    
    optimizer = AdamW([
        {'params': backbone_params, 'lr': LR * 0.1},
        {'params': mc_aux_params, 'lr': LR * 0.5},
        {'params': memory_params, 'lr': LR * 0.5},
        {'params': head_params, 'lr': LR}
    ], weight_decay=0.01)
    
    num_training_steps = len(train_loader) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_training_steps // 10,
        num_training_steps=num_training_steps
    )
    
    print(f"\nStarting Aux + M-C + CEMN training...")
    
    best_f1 = 0.0
    best_threshold = 0.5
    patience = 4
    patience_counter = 0
    
    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()
            loss, _, _ = model(
                batch['input_ids'].to(device),
                batch['attention_mask'].to(device),
                batch['kw_input_ids'].to(device),
                batch['kw_attention_mask'].to(device),
                batch['loc_input_ids'].to(device),
                batch['loc_attention_mask'].to(device),
                batch['labels'].to(device),
                batch['hier_label'].to(device)
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
                loss, logits, _ = model(
                    batch['input_ids'].to(device),
                    batch['attention_mask'].to(device),
                    batch['kw_input_ids'].to(device),
                    batch['kw_attention_mask'].to(device),
                    batch['loc_input_ids'].to(device),
                    batch['loc_attention_mask'].to(device),
                    batch['labels'].to(device),
                    batch['hier_label'].to(device)
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
        
        preds_05 = (all_logits >= 0.5).astype(int)
        f1_05 = f1_score(all_labels, preds_05)
        acc_05 = accuracy_score(all_labels, preds_05)
        
        print(f"Epoch {epoch}:")
        print(f"  Loss - Train: {avg_loss:.4f}, Val: {val_loss:.4f}")
        print(f"  F1 @ 0.5: {f1_05:.4f}, Acc: {acc_05:.4f}")
        print(f"  Best F1: {best_thr_f1:.4f} @ threshold {best_thr:.3f}")
        
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
                path = f"{SAVE_DIR}/auxmc_cemn_f1_{best_thr_f1:.4f}.pth"
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
        'model_type': 'auxmc_cemn'
    }
    with open(f"{SAVE_DIR}/auxmc_cemn_results.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()