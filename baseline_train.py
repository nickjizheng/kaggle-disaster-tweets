import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import ElectraTokenizer, ElectraModel, get_cosine_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import json
import warnings
import random
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# Enhanced mislabeled corrections - expanded list
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

# Load and preprocess data
print("Loading data...")
# The CSV has no header and 5 columns: id, keyword, location, text, target
df = pd.read_csv('kaggle/train.csv', header=None, names=['id', 'keyword', 'location', 'text', 'target'])

print("Relabeling misidentified samples...")
df['target_relabeled'] = df['target']
for txt, lbl in MISLABELED_CORRECTIONS:
    df.loc[df['text'] == txt, 'target_relabeled'] = lbl

# No text cleaning - use original text
print("Using original text without cleaning...")

df['keyword']  = df['keyword'].fillna('no_keyword')
df['location'] = df['location'].fillna('no_location')

# Convert empty strings to default values
df['keyword'] = df['keyword'].replace('', 'no_keyword')
df['location'] = df['location'].replace('', 'no_location')

# Check class distribution
print(f"\nClass distribution:")
print(f"Class 0 (Not Disaster): {sum(df['target_relabeled'] == 0)} ({100*sum(df['target_relabeled'] == 0)/len(df):.1f}%)")
print(f"Class 1 (Disaster): {sum(df['target_relabeled'] == 1)} ({100*sum(df['target_relabeled'] == 1)/len(df):.1f}%)")

# Enhanced Dataset with data augmentation
class AdvancedTweetDataset(Dataset):
    def __init__(self, texts, labels, keywords, locations, tokenizer, max_len=128, augment=False):
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
        """Simple data augmentation techniques"""
        if not self.augment or random.random() > 0.3:
            return text
        
        # Random synonym replacement or word dropout
        words = text.split()
        if len(words) > 3:
            # Randomly drop 1 word (10% chance)
            if random.random() < 0.1:
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
        
        # Smarter metadata integration
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

# Multi-Sample Dropout for regularization
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
        
        # Average predictions from multiple dropout rates
        logits = torch.stack(logits_list, dim=0).mean(dim=0)
        return logits

# Simplified model architecture
class SimplifiedElectraModel(nn.Module):
    def __init__(self, backbone: ElectraModel, dropout_probs=[0.1, 0.2, 0.3]):
        super().__init__()
        self.electra = backbone
        
        # Freeze embeddings but keep encoder trainable
        for param in self.electra.embeddings.parameters():
            param.requires_grad = False
        
        hs = backbone.config.hidden_size
        
        # Simple feature extraction using CLS token and mean pooling
        self.feature_extractor = nn.Sequential(
            nn.Linear(hs * 2, hs),  # CLS + Mean pooling
            nn.GELU(),
            nn.LayerNorm(hs),
            nn.Dropout(0.1),
            nn.Linear(hs, hs // 2),
            nn.GELU(),
            nn.LayerNorm(hs // 2),
        )
        
        # Multi-sample dropout classifier
        self.classifier = MultiSampleDropout(hs // 2, 2, dropout_probs)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # CLS token embedding
        cls_emb = hidden_states[:, 0, :]
        
        # Mean pooling with attention mask
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        mean_emb = sum_embeddings / sum_mask
        
        # Combine CLS and mean pooling
        combined = torch.cat([cls_emb, mean_emb], dim=-1)
        features = self.feature_extractor(combined)
        
        # Main classification
        logits = self.classifier(features)
        
        if labels is not None:
            # Loss with label smoothing
            loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
            return loss, logits
        
        return logits

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load ELECTRA
ELECTRA_PATH = '/root/autodl-fs/ELECTRA'
tokenizer = ElectraTokenizer.from_pretrained(ELECTRA_PATH, local_files_only=True)
backbone = ElectraModel.from_pretrained(ELECTRA_PATH, local_files_only=True)

# Split data with stratification
train_df, val_df = train_test_split(
    df, test_size=0.15, random_state=42, stratify=df['target_relabeled']  # Smaller val set
)

print(f"\nData split:")
print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")

# Create datasets with augmentation for training - using original text instead of cleaned
train_ds = AdvancedTweetDataset(
    train_df['text'].tolist(),  # Using original text instead of text_cleaned
    train_df['target_relabeled'].tolist(),
    train_df['keyword'].tolist(),
    train_df['location'].tolist(),
    tokenizer,
    max_len=140,  # Slightly longer for better context
    augment=True
)

val_ds = AdvancedTweetDataset(
    val_df['text'].tolist(),  # Using original text instead of text_cleaned
    val_df['target_relabeled'].tolist(),
    val_df['keyword'].tolist(),
    val_df['location'].tolist(),
    tokenizer,
    max_len=140,
    augment=False
)

# Advanced training parameters
EPOCHS = 24
BATCH = 12  # Smaller batch for better gradient estimates
LR = 8e-5   # Even lower LR
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

# Model and optimizer
model = SimplifiedElectraModel(backbone).to(device)

# Different learning rates for different parts
backbone_params = list(model.electra.parameters())
head_params = list(model.feature_extractor.parameters()) + list(model.classifier.parameters())

optimizer = AdamW([
    {'params': backbone_params, 'lr': LR * 0.1},  # Lower LR for backbone
    {'params': head_params, 'lr': LR}             # Higher LR for head
], weight_decay=0.01)

# Cosine annealing with warm restarts
num_training_steps = len(train_loader) * EPOCHS
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_training_steps // 10,
    num_training_steps=num_training_steps
)

print(f"\nStarting simplified training...")
print(f"Epochs: {EPOCHS}, Batch size: {BATCH}, Learning rate: {LR}")

best_f1 = 0.0
best_threshold = 0.5
patience = 4
patience_counter = 0

for epoch in range(1, EPOCHS+1):
    # Training
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
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
        for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Validation")):
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

    # Find optimal threshold using F1 score
    best_thr, best_thr_f1 = 0.5, 0.0
    for thr in np.linspace(0.1, 0.9, 161):  # More granular search
        preds = (all_logits >= thr).astype(int)
        f1 = f1_score(all_labels, preds, zero_division=0)
        if f1 > best_thr_f1:
            best_thr_f1, best_thr = f1, thr

    # Metrics at 0.5 and optimal threshold
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
        
        import shutil
        free_space = shutil.disk_usage(SAVE_DIR).free / (1024**3)
        if free_space > 1.0:
            try:
                ckpt = {
                    'model_state_dict': model.state_dict(),
                    'threshold': best_thr,
                    'f1_score': best_thr_f1,
                    'epoch': epoch
                }
                path = f"{SAVE_DIR}/simplified_model_f1_{best_thr_f1:.4f}.pth"
                torch.save(ckpt, path)
                print(f"  >>> Saved: {path}")
            except Exception as e:
                print(f"  Save failed: {e}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"  Early stopping after {patience} epochs")
            break
    
    print("-" * 60)

print(f"\nTraining completed! Best F1: {best_f1:.4f} @ threshold {best_threshold:.3f}")

# Save results
results = {'best_f1': float(best_f1), 'best_threshold': float(best_threshold)}
with open(f"{SAVE_DIR}/simplified_results.json", "w") as f:
    json.dump(results, f)