import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ElectraTokenizer, ElectraModel
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import warnings
import os
warnings.filterwarnings('ignore')

# Reproduce the EXACT same model architecture from combined training
class MultiSampleDropout(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout_probs=[0.1, 0.2, 0.3, 0.4, 0.5]):
        super().__init__()
        self.dropout_probs = dropout_probs
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in dropout_probs])
        self.classifier = nn.Linear(hidden_size, num_labels)
    
    def forward(self, x):
        # During evaluation, we still want to use all dropouts for better ensemble
        if self.training:
            logits_list = []
            for dropout in self.dropouts:
                logits_list.append(self.classifier(dropout(x)))
        else:
            # During eval, still use dropout for ensemble effect
            logits_list = []
            for dropout in self.dropouts:
                dropout.eval()  # But set to eval mode
                logits_list.append(self.classifier(dropout(x)))
        
        logits = torch.stack(logits_list, dim=0).mean(dim=0)
        return logits

# COMBINED Model with both Multi-Channel and Hierarchical features
class CombinedMultiChannelHierarchicalModel(nn.Module):
    def __init__(self, backbone: ElectraModel, dropout_probs=[0.1, 0.2, 0.3]):
        super().__init__()
        # Single shared ELECTRA encoder
        self.electra = backbone
        
        # Freeze embeddings
        for param in self.electra.embeddings.parameters():
            param.requires_grad = False
        
        hs = backbone.config.hidden_size
        
        # Lightweight encoders for auxiliary channels
        self.kw_encoder = nn.Sequential(
            nn.Linear(hs, hs // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hs // 2, hs)
        )
        
        self.loc_encoder = nn.Sequential(
            nn.Linear(hs, hs // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hs // 2, hs)
        )
        
        # Multi-head attention pooling
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=hs, 
            num_heads=8, 
            batch_first=True,
            dropout=0.1
        )
        
        # Cross-modal attention for keyword and location
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=hs,
            num_heads=4,
            batch_first=True,
            dropout=0.1
        )
        
        # Feature fusion with gating
        self.fusion_gate = nn.Sequential(
            nn.Linear(hs * 3, hs),
            nn.Sigmoid()
        )
        
        # Combined feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(hs * 2, hs * 2),  # Main features + gated auxiliary
            nn.GELU(),
            nn.LayerNorm(hs * 2),
            nn.Dropout(0.1),
            nn.Linear(hs * 2, hs),
            nn.GELU(),
            nn.LayerNorm(hs),
        )
        
        # Main classifier with multi-sample dropout
        self.classifier = MultiSampleDropout(hs, 2, dropout_probs)
        
        # Hierarchical head
        self.hier_head = nn.Sequential(
            nn.Linear(hs, hs // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hs // 2, 3)  # 3 classes: no_kw, disaster_kw, other_kw
        )

    def forward(self, input_ids, attention_mask, kw_input_ids, kw_attention_mask,
                loc_input_ids, loc_attention_mask, labels=None, hier_labels=None):
        
        # Main encoding (text with metadata)
        outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Multiple pooling strategies
        cls_emb = hidden_states[:, 0, :]
        
        # Attention-based pooling
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
        
        # Combine main features
        main_features = torch.cat([cls_emb, attn_emb, mean_emb], dim=-1)
        
        # Process auxiliary channels
        kw_outputs = self.electra(input_ids=kw_input_ids, attention_mask=kw_attention_mask)
        loc_outputs = self.electra(input_ids=loc_input_ids, attention_mask=loc_attention_mask)
        
        kw_features = self.kw_encoder(kw_outputs.last_hidden_state[:, 0, :])
        loc_features = self.loc_encoder(loc_outputs.last_hidden_state[:, 0, :])
        
        # Cross-modal attention between auxiliary features and main CLS
        aux_features = torch.stack([kw_features, loc_features], dim=1)
        cross_attended, _ = self.cross_modal_attention(
            cls_emb.unsqueeze(1), aux_features, aux_features
        )
        cross_attended = cross_attended.squeeze(1)
        
        # Gated fusion
        gate = self.fusion_gate(main_features)
        fused_features = torch.cat([
            main_features.mean(dim=-1, keepdim=True).expand(-1, cls_emb.size(-1)),
            cross_attended * gate + cls_emb * (1 - gate)
        ], dim=-1)
        
        # Extract final features
        features = self.feature_extractor(fused_features)
        
        # Classification heads
        logits = self.classifier(features)
        hier_logits = self.hier_head(features)
        
        if labels is not None:
            # Main task loss
            main_loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
            
            # Hierarchical loss
            if hier_labels is not None:
                hier_loss = F.cross_entropy(hier_logits, hier_labels)
                total_loss = main_loss + 0.1 * hier_loss
            else:
                total_loss = main_loss
            
            return total_loss, logits, hier_logits
        
        return logits, hier_logits

# Test Dataset class with multi-channel encoding
class TestTweetDataset(Dataset):
    def __init__(self, texts, keywords, locations, tokenizer, max_len=140):
        self.texts = texts
        self.keywords = keywords
        self.locations = locations
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        txt = self.texts[idx]
        kw = self.keywords[idx]
        loc = self.locations[idx]
        
        # Create enhanced text with metadata (same as training)
        metadata_parts = []
        if kw != 'no_keyword':
            metadata_parts.append(f"[KEYWORD: {kw}]")
        if loc != 'no_location':
            metadata_parts.append(f"[LOCATION: {loc}]")
        
        if metadata_parts:
            enhanced = f"{' '.join(metadata_parts)} {txt}"
        else:
            enhanced = txt

        # Main encoding
        enc = self.tokenizer.encode_plus(
            enhanced,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Also encode keyword and location separately (multi-channel)
        kw_enc = self.tokenizer.encode_plus(
            kw if kw != 'no_keyword' else '',
            add_special_tokens=True,
            max_length=32,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        loc_enc = self.tokenizer.encode_plus(
            loc if loc != 'no_location' else '',
            add_special_tokens=True,
            max_length=64,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'kw_input_ids': kw_enc['input_ids'].squeeze(0),
            'kw_attention_mask': kw_enc['attention_mask'].squeeze(0),
            'loc_input_ids': loc_enc['input_ids'].squeeze(0),
            'loc_attention_mask': loc_enc['attention_mask'].squeeze(0)
        }

def load_model_and_threshold(model_path, backbone, device):
    """Load the trained model and its optimal threshold"""
    model = CombinedMultiChannelHierarchicalModel(backbone).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get optimal threshold if saved
    threshold = checkpoint.get('threshold', 0.5)
    f1_score = checkpoint.get('f1_score', None)
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    if f1_score:
        print(f"Validation F1 score: {f1_score:.4f}")
    print(f"Saved threshold in model: {threshold:.4f}")
    
    return model, threshold

def predict(model, test_loader, device, threshold=0.5):
    """Make predictions on test data"""
    model.eval()
    predictions = []
    probabilities = []
    
    # Hierarchical predictions
    hier_predictions = []
    hier_names = ['no_keyword', 'disaster_keyword', 'other_keyword']
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            # Get all inputs for multi-channel model
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            kw_input_ids = batch['kw_input_ids'].to(device)
            kw_attention_mask = batch['kw_attention_mask'].to(device)
            loc_input_ids = batch['loc_input_ids'].to(device)
            loc_attention_mask = batch['loc_attention_mask'].to(device)
            
            # Forward pass with all inputs
            main_logits, hier_logits = model(
                input_ids, attention_mask,
                kw_input_ids, kw_attention_mask,
                loc_input_ids, loc_attention_mask
            )
            
            # Main task predictions
            probs = torch.softmax(main_logits, dim=1)[:, 1].cpu().numpy()
            preds = (probs >= threshold).astype(int)
            
            # Hierarchical predictions
            hier_preds = torch.argmax(hier_logits, dim=1).cpu().numpy()
            
            predictions.extend(preds)
            probabilities.extend(probs)
            hier_predictions.extend(hier_preds)
    
    return predictions, probabilities, hier_predictions, hier_names

def main():
    # ============================================
    # MANUAL CONFIGURATION - EDIT THESE PATHS
    # ============================================
    MODEL_PATH = '/root/autodl-fs/model/combined_multichannel_hierarchical_f1_0.8XXX.pth'  # <-- CHANGE THIS TO YOUR MODEL
    THRESHOLD = None # <-- CHANGE THIS IF YOU WANT A DIFFERENT THRESHOLD (or None to use saved threshold)
    TEST_FILE = 'kaggle/test.csv'
    ELECTRA_PATH = '/root/autodl-fs/ELECTRA'
    OUTPUT_DIR = 'kaggle'  # Where to save predictions
    
    # ============================================
    # END OF MANUAL CONFIGURATION
    # ============================================
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found: {MODEL_PATH}")
        print("\nAvailable model files in /root/autodl-fs/model/:")
        model_dir = '/root/autodl-fs/model/'
        if os.path.exists(model_dir):
            for f in sorted(os.listdir(model_dir)):
                if f.endswith('.pth'):
                    print(f"  - {f}")
        return
    
    print(f"Using model: {MODEL_PATH}")
    
    # Load test data
    print("\nLoading test data...")
    if not os.path.exists(TEST_FILE):
        print(f"Error: {TEST_FILE} not found!")
        return
    
    # Try to load test.csv with different formats
    try:
        # First try with headers
        test_df = pd.read_csv(TEST_FILE)
        if 'text' not in test_df.columns:
            raise ValueError("No 'text' column found")
        print("Loaded test.csv with headers")
    except:
        # Try without headers
        test_df = pd.read_csv(TEST_FILE, header=None)
        if len(test_df.columns) == 4:
            test_df.columns = ['id', 'keyword', 'location', 'text']
            print("Loaded test.csv without headers (4 columns)")
        else:
            print(f"Error: Unexpected number of columns: {len(test_df.columns)}")
            print("Expected format: id, keyword, location, text")
            return
    
    print(f"Test dataset shape: {test_df.shape}")
    
    # Handle missing values
    test_df['keyword'] = test_df['keyword'].fillna('no_keyword').astype(str).replace('', 'no_keyword')
    test_df['location'] = test_df['location'].fillna('no_location').astype(str).replace('', 'no_location')
    
    print("\nSample test data:")
    print(test_df.head())
    
    # Load ELECTRA tokenizer and backbone
    print("\nLoading ELECTRA model...")
    if not os.path.exists(ELECTRA_PATH):
        print(f"Error: ELECTRA model not found at {ELECTRA_PATH}")
        return
    
    tokenizer = ElectraTokenizer.from_pretrained(ELECTRA_PATH, local_files_only=True)
    backbone = ElectraModel.from_pretrained(ELECTRA_PATH, local_files_only=True)
    
    # Load model and threshold
    print("\nLoading trained model...")
    model, saved_threshold = load_model_and_threshold(MODEL_PATH, backbone, device)
    
    # Use manual threshold if specified, otherwise use saved threshold
    if THRESHOLD is not None:
        threshold = THRESHOLD
        print(f"Using manually specified threshold: {threshold:.4f}")
    else:
        threshold = saved_threshold
        print(f"Using saved threshold from training: {threshold:.4f}")
    
    # Create test dataset with multi-channel encoding
    test_dataset = TestTweetDataset(
        test_df['text'].tolist(),
        test_df['keyword'].tolist(),
        test_df['location'].tolist(),
        tokenizer,
        max_len=140
    )
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Make predictions
    print("\nMaking predictions...")
    predictions, probabilities, hier_predictions, hier_names = predict(
        model, test_loader, device, threshold
    )
    
    # Create submission file
    submission = pd.DataFrame({
        'id': test_df['id'],
        'target': predictions
    })
    
    # Create detailed results
    detailed_results = pd.DataFrame({
        'id': test_df['id'],
        'text': test_df['text'],
        'keyword': test_df['keyword'],
        'location': test_df['location'],
        'probability': probabilities,
        'prediction': predictions,
        'hierarchical_class': [hier_names[idx] for idx in hier_predictions]
    })
    
    # Save results
    submission_path = os.path.join(OUTPUT_DIR, 'submission.csv')
    detailed_path = os.path.join(OUTPUT_DIR, 'detailed_predictions.csv')
    
    submission.to_csv(submission_path, index=False)
    detailed_results.to_csv(detailed_path, index=False)
    
    print(f"\n✅ Submission saved to: {submission_path}")
    print(f"✅ Detailed results saved to: {detailed_path}")
    
    # Show prediction statistics
    print(f"\n" + "="*50)
    print("PREDICTION STATISTICS")
    print("="*50)
    print(f"Total samples: {len(predictions)}")
    print(f"Predicted as disaster (1): {sum(predictions)} ({100*sum(predictions)/len(predictions):.1f}%)")
    print(f"Predicted as not disaster (0): {len(predictions)-sum(predictions)} ({100*(len(predictions)-sum(predictions))/len(predictions):.1f}%)")
    print(f"\nProbability statistics:")
    print(f"  Mean: {np.mean(probabilities):.4f}")
    print(f"  Std: {np.std(probabilities):.4f}")
    print(f"  Min: {np.min(probabilities):.4f}")
    print(f"  Max: {np.max(probabilities):.4f}")
    print(f"  Threshold used: {threshold:.4f}")
    
    # Show hierarchical prediction statistics
    print(f"\n" + "="*50)
    print("HIERARCHICAL PREDICTION STATISTICS")
    print("="*50)
    for i, name in enumerate(hier_names):
        count = sum([1 for p in hier_predictions if p == i])
        print(f"{name}: {count} ({100*count/len(hier_predictions):.1f}%)")
    
    # Cross-tabulation: hierarchical class vs main prediction
    print(f"\n" + "="*50)
    print("CROSS-TABULATION: Hierarchical Class vs Main Prediction")
    print("="*50)
    for i, hier_name in enumerate(hier_names):
        hier_mask = [idx for idx, p in enumerate(hier_predictions) if p == i]
        if hier_mask:
            disaster_count = sum([predictions[idx] for idx in hier_mask])
            total_count = len(hier_mask)
            print(f"{hier_name}:")
            print(f"  Total: {total_count}")
            print(f"  Predicted disaster: {disaster_count} ({100*disaster_count/total_count:.1f}%)")
            print(f"  Predicted non-disaster: {total_count-disaster_count} ({100*(total_count-disaster_count)/total_count:.1f}%)")
    
    # Show sample predictions
    print(f"\n" + "="*50)
    print("SAMPLE PREDICTIONS")
    print("="*50)
    n_samples = min(10, len(detailed_results))
    for i in range(n_samples):
        row = detailed_results.iloc[i]
        print(f"\n[{i+1}] ID: {row['id']}")
        print(f"Text: {row['text'][:100]}{'...' if len(row['text']) > 100 else ''}")
        print(f"Keyword: {row['keyword']}")
        print(f"Prediction: {'🔴 DISASTER' if row['prediction'] == 1 else '🟢 NOT DISASTER'} (prob: {row['probability']:.4f})")
        print(f"Hierarchical: {row['hierarchical_class']}")
        print("-" * 40)
    
    print(f"\n✅ Testing completed successfully!")
    print(f"Submission file ready for Kaggle: {submission_path}")

if __name__ == "__main__":
    main()