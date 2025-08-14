import os
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from albumentations import Compose, Normalize, Resize
from albumentations import RandomResizedCrop, HorizontalFlip, VerticalFlip, RandomBrightnessContrast
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score
from torchvision import models
from sklearn.model_selection import train_test_split
from logging import getLogger, DEBUG, FileHandler, Formatter, StreamHandler
from tqdm import tqdm
import numpy as np
from PIL import Image
import time
import csv
from collections import Counter

from utils import get_subset

# ------------------------------
# Utilities
# ------------------------------

def ensure_folder(folder):
    if not os.path.exists(folder):
        print(f"Folder '{folder}' does not exist. Creating...")
        os.makedirs(folder)

def seed_torch(seed=777):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def initialize_csv_logger(file_path):
    header = [
        "epoch","time_sec",
        "train_loss","train_acc","train_f1_macro","train_f1_weighted",
        "val_loss","val_acc","val_f1_macro","val_f1_weighted","lr"
    ]
    with open(file_path, mode='w', newline='') as csv_file:
        csv.writer(csv_file).writerow(header)

def log_epoch_to_csv(file_path, row):
    with open(file_path, mode='a', newline='') as csv_file:
        csv.writer(csv_file).writerow(row)

def get_transforms(data):
    width, height = 224, 224
    if data == 'train':
        return Compose([
            RandomResizedCrop((width, height), scale=(0.8, 1.0)),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            RandomBrightnessContrast(p=0.2),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    elif data == 'valid':
        return Compose([
            Resize(width, height),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        raise ValueError("Unknown data mode (only 'train' or 'valid').")

# ------------------------------
# Metadata preprocessing (robust to missing)
# ------------------------------

OUT_DIR = 'processed_metadata'

def _safe_one_hot_with_unknown(series: pd.Series, name: str):
    # categories seen in entire df (excluding NaN) + UNKNOWN at the end
    cats = pd.Categorical(series)
    categories = [c for c in cats.categories]  # no NaN here
    unknown_idx = len(categories)
    codes = pd.Categorical(series, categories=categories).codes  # -1 for NaN
    codes = np.where(codes == -1, unknown_idx, codes)  # map NaN -> UNKNOWN

    num_classes = len(categories) + 1  # include UNKNOWN
    codes_t = torch.tensor(codes, dtype=torch.long)
    onehot = torch.nn.functional.one_hot(codes_t, num_classes=num_classes).float()

    # add a known/unknown bit as the last column (+1 feature)
    known_bit = torch.tensor(series.notna().astype(np.float32).values).unsqueeze(1)
    enriched = torch.cat([onehot, known_bit], dim=1)
    torch.save(enriched, os.path.join(OUT_DIR, f'{name}.pt'))
    # also store dims
    torch.save(torch.tensor([num_classes + 1], dtype=torch.long), os.path.join(OUT_DIR, f'{name}_dim.pt'))

def preprocess_metadata():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv('data/metadata.csv')
    train_df = get_subset(df, 'train')  # used for min/max normalization

    # Categorical with UNKNOWN + known bit
    _safe_one_hot_with_unknown(df['Habitat'], 'Habitat')
    _safe_one_hot_with_unknown(df['Substrate'], 'Substrate')

    # Geo: min-max on TRAIN ONLY, scale to [-1,1], fill missing with 0, add known bit (shared for lat/lon)
    lat_min, lat_max = train_df['Latitude'].min(), train_df['Latitude'].max()
    lon_min, lon_max = train_df['Longitude'].min(), train_df['Longitude'].max()

    lat = df['Latitude']
    lon = df['Longitude']
    lat_s = (((lat - lat_min) / (lat_max - lat_min)) - 0.5) * 2.0
    lon_s = (((lon - lon_min) / (lon_max - lon_min)) - 0.5) * 2.0

    # known bit: both present
    geo_known = (~lat.isna() & ~lon.isna()).astype(np.float32)
    lat_s = lat_s.fillna(0.0)
    lon_s = lon_s.fillna(0.0)

    geo = torch.tensor(np.stack([lat_s.values, lon_s.values, geo_known.values], axis=1), dtype=torch.float32)
    torch.save(geo, os.path.join(OUT_DIR, 'geo.pt'))
    torch.save(torch.tensor([3], dtype=torch.long), os.path.join(OUT_DIR, 'geo_dim.pt'))

    # eventDate -> cyc features + year trend, fill with 0, add known bit
    event_date = pd.to_datetime(df['eventDate'], errors='coerce')
    known = event_date.notna().astype(np.float32)

    month = event_date.dt.month.fillna(0.0)
    dayofyear = event_date.dt.dayofyear.fillna(0.0)
    week = event_date.dt.isocalendar().get('week').astype('float').fillna(0.0)

    month_sin = np.sin(2 * np.pi * month / 12.0)
    month_cos = np.cos(2 * np.pi * month / 12.0)
    doy_sin = np.sin(2 * np.pi * dayofyear / 365.25)
    doy_cos = np.cos(2 * np.pi * dayofyear / 365.25)
    week_sin = np.sin(2 * np.pi * week / 52.0)
    week_cos = np.cos(2 * np.pi * week / 52.0)

    # clip year range as before (but robust to NaT)
    y = event_date.dt.year
    y = y.fillna(y.median() if y.notna().any() else 2000)
    max_year, min_year = 2020, 1985
    year_scaled = 2 * (y - min_year) / (max_year - min_year) - 1

    time_feats = np.stack([month_sin, month_cos, doy_sin, doy_cos, week_sin, week_cos, year_scaled, known], axis=1).astype(np.float32)
    time_data = torch.tensor(time_feats, dtype=torch.float32)
    torch.save(time_data, os.path.join(OUT_DIR, 'eventDate.pt'))
    torch.save(torch.tensor([8], dtype=torch.long), os.path.join(OUT_DIR, 'eventDate_dim.pt'))

def load_metadata_and_dims():
    cols = ['Habitat', 'Substrate', 'geo', 'eventDate']
    data, dims = {}, {}
    for col in cols:
        path = os.path.join(OUT_DIR, f'{col}.pt')
        dimp = os.path.join(OUT_DIR, f'{col}_dim.pt')
        if not os.path.exists(path):
            preprocess_metadata()
        data[col] = torch.load(path)
        dims[col] = int(torch.load(dimp).item())
    return data, dims

# ------------------------------
# Dataset
# ------------------------------

class FungiDataset(Dataset):
    def __init__(self, df, path, transform=None):
        self.df = df.copy()
        self.transform = transform
        self.path = path

        # Keep global row indices to align with precomputed tensors
        self.row_idx = self.df.index.values

        (meta, dims) = load_metadata_and_dims()
        self.meta = meta
        self.dims = dims

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        ridx = self.row_idx[idx]
        file_path = self.df['filename_index'].values[idx]
        label = self.df['taxonID_index'].values[idx]
        label = -1 if pd.isnull(label) else int(label)

        with Image.open(os.path.join(self.path, file_path)) as img:
            image = img.convert('RGB')
        image = np.array(image)
        if self.transform:
            image = self.transform(image=image)['image']

        habitat = self.meta['Habitat'][ridx]
        substrate = self.meta['Substrate'][ridx]
        geo = self.meta['geo'][ridx]
        eventDate = self.meta['eventDate'][ridx]

        return image, habitat, substrate, geo, eventDate, label, file_path

# ------------------------------
# Model: CMSE-style fusion for Image + Metadata
# ------------------------------

class MBFE(nn.Module):
    """Multi-Branch Feature Extraction with 1x1, 3x3, 5x5 convs."""
    def __init__(self, in_ch, out_ch_each=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch_each, kernel_size=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(in_ch, out_ch_each, kernel_size=3, padding=1, bias=False)
        self.conv5 = nn.Conv2d(in_ch, out_ch_each, kernel_size=5, padding=2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch_each * 3)
        self.act = nn.GELU()
    def forward(self, x):
        x = torch.cat([self.conv1(x), self.conv3(x), self.conv5(x)], dim=1)
        return self.act(self.bn(x))

class SemanticProjector(nn.Module):
    """Project flattened spatial features into K semantic tokens via attention pooling."""
    def __init__(self, in_dim, K=64):
        super().__init__()
        self.W = nn.Linear(in_dim, K, bias=False)  # Gaussian-initialized in spirit; normal init works
    def forward(self, x_bhwc):
        # x: (B, H*W, C)
        logits = self.W(x_bhwc)          # (B, N, K)
        attn = torch.softmax(logits, dim=1)  # softmax over positions
        tokens = torch.einsum('bnk,bnc->bkc', attn, x_bhwc)  # (B, K, C)
        return tokens  # semantic tokens

class EmbeddingNet(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.1):
        super().__init__()
        self.module = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, out_features),
            nn.GELU(),
        )
    def forward(self, x): return self.module(x)

def gaussian_wasserstein_loss(x_tokens: torch.Tensor, m_tokens: torch.Tensor):
    """
    x_tokens: (B, K, D) aligned image tokens
    m_tokens: (B, K, D) metadata tokens
    Compute batchwise 2-Wasserstein (Bures) distance for diagonal Gaussians:
      W2^2 = ||mu_x - mu_m||^2 + ||sqrt(var_x) - sqrt(var_m)||^2
    where mu and var are computed across the token axis (dim=1).
    """
    # means and variances over the token dimension (K)
    mu_x = x_tokens.mean(dim=1)                                   # (B, D)
    mu_m = m_tokens.mean(dim=1)                                   # (B, D)
    var_x = x_tokens.var(dim=1, unbiased=False) + 1e-6            # (B, D)
    var_m = m_tokens.var(dim=1, unbiased=False) + 1e-6            # (B, D)

    mean_term = torch.sum((mu_x - mu_m) ** 2, dim=-1)             # (B,)
    cov_term  = torch.sum((torch.sqrt(var_x) - torch.sqrt(var_m)) ** 2, dim=-1)  # (B,)
    return (mean_term + cov_term).mean()

class CMSEMixNet(nn.Module):
    """
    CMSE-inspired fusion for RGB image + tabular metadata.
    """
    def __init__(self, num_classes, dims, token_K=64, meta_embed_dims=(32,128,64,32)):
        super().__init__()

        # ---- EfficientNet-B0 with modern weights API + offline fallback ----
        try:
            from torchvision.models import EfficientNet_B0_Weights
            self.backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        except Exception:
            # no internet / older torchvision: proceed without ImageNet weights
            self.backbone = models.efficientnet_b0(weights=None)

        # we'll tap spatial features directly
        self.backbone.classifier = nn.Identity()

        # Robustly detect feature channels (works across torchvision versions)
        with torch.no_grad():
            self.backbone.eval()
            dummy = torch.zeros(1, 3, 224, 224, dtype=torch.float32)
            feat_ch = self.backbone.features(dummy).shape[1]
        self.backbone.train()

        # MBFE + projector
        self.mbfe = MBFE(in_ch=feat_ch, out_ch_each=64)   # -> 192 channels
        self.projector = SemanticProjector(in_dim=192, K=token_K)

        # Align image token channel dimension (192) to meta token dim (64) for GW loss
        self.img_to_meta = nn.Linear(192, 64, bias=False)
        self.img_token_norm = nn.LayerNorm(64)

        # Metadata embeddings (dims already include known-bit augmentation)
        h_dim = dims['Habitat']
        s_dim = dims['Substrate']
        g_dim = dims['geo']
        t_dim = dims['eventDate']

        self.temporal_embedding  = EmbeddingNet(t_dim, meta_embed_dims[0])
        self.habitat_embedding   = EmbeddingNet(h_dim, meta_embed_dims[1])
        self.substrate_embedding = EmbeddingNet(s_dim, meta_embed_dims[2])
        self.geo_embedding       = EmbeddingNet(g_dim, meta_embed_dims[3])

        meta_total = sum(meta_embed_dims)
        self.meta_to_tokens = nn.Linear(meta_total, token_K * 64, bias=False)  # K tokens of dim 64
        self.meta_norm = nn.LayerNorm(64)

        # FiLM from meta → image branches
        self.film_gamma = nn.Linear(64, 192)
        self.film_beta  = nn.Linear(64, 192)

        # Head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.cls = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(192 + meta_total, num_classes)
        )

    def forward_backbone_features(self, x):
        # Extract spatial features from EfficientNet
        x = self.backbone.features(x)  # (B, C, H, W)
        return x

    def forward(self, image, habitat, substrate, geo, eventDate, return_tokens=False):
        # ----- Image path -----
        fmap = self.forward_backbone_features(image)          # (B,C,H,W)
        f = self.mbfe(fmap)                                   # (B,192,H,W)

        B, C, H, W = f.shape
        f_flat = f.permute(0,2,3,1).reshape(B, H*W, C)        # (B,N,192)
        img_tokens = self.projector(f_flat)                   # (B,K,192)
        img_tokens_aligned = self.img_token_norm(self.img_to_meta(img_tokens))  # (B,K,64)

        # ----- Metadata path -----
        t_emb  = self.temporal_embedding(eventDate)
        h_emb  = self.habitat_embedding(habitat)
        s_emb  = self.substrate_embedding(substrate)
        g_emb  = self.geo_embedding(geo)
        meta_vec = torch.cat([t_emb, h_emb, s_emb, g_emb], dim=1)  # (B, meta_total)

        meta_tokens = self.meta_to_tokens(meta_vec).view(B, -1, 64)  # (B,K,64)
        meta_tokens = self.meta_norm(meta_tokens)

        # ----- Residual fusion via FiLM (use pooled meta token) -----
        meta_global = meta_tokens.mean(dim=1)                 # (B,64)
        gamma = torch.sigmoid(self.film_gamma(meta_global)).unsqueeze(-1).unsqueeze(-1)  # (B,192,1,1)
        beta  = self.film_beta(meta_global).unsqueeze(-1).unsqueeze(-1)                  # (B,192,1,1)
        f_fused = f * gamma + beta                            # FiLM
        f_out = f + f_fused                                   # residual fusion

        # ----- Classification -----
        pooled = self.pool(f_out).view(B, C)                  # (B,192)
        logits = self.cls(torch.cat([pooled, meta_vec], dim=1))

        if return_tokens:
            # Return aligned image tokens to ensure GW loss gets matching channel dims
            return logits, img_tokens_aligned, meta_tokens
        return logits

# ------------------------------
# Training / Eval
# ------------------------------

BATCH_SIZE = 128
NUM_CLASSES = 183
ALPHA_CE = 0.6   # weight for CE
BETA_GW  = 0.4   # weight for Gaussian-Wasserstein alignment

def _compute_class_weights(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes)
    counts[counts == 0] = 1
    inv = 1.0 / counts
    weights = inv / inv.mean()
    return torch.tensor(weights, dtype=torch.float32)

def train_fungi_network(data_file, image_path, checkpoint_dir):
    ensure_folder(checkpoint_dir)
    csv_file_path = os.path.join(checkpoint_dir, 'train.csv')
    initialize_csv_logger(csv_file_path)

    df = pd.read_csv(data_file)
    train_df = df[df['filename_index'].str.startswith('fungi_train')].copy()
    train_df, val_df = train_test_split(
        train_df, test_size=0.2, random_state=42,
        stratify=train_df['taxonID_index'], shuffle=True
    )
    # reset indexes so our dataset keeps original row indices via .index
    train_df = train_df.sort_index()
    val_df = val_df.sort_index()
    print('Training size', len(train_df))
    print('Validation size', len(val_df))

    train_dataset = FungiDataset(train_df, image_path, transform=get_transforms('train'))
    valid_dataset = FungiDataset(val_df, image_path, transform=get_transforms('valid'))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    meta_dims = train_dataset.dims
    model = CMSEMixNet(num_classes=NUM_CLASSES, dims=meta_dims).to(device)

    # Class-weighted CE to help Macro-F1
    y_train = train_df['taxonID_index'].dropna().astype(int).values
    class_weights = _compute_class_weights(y_train, NUM_CLASSES).to(device)
    ce = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=2, eps=1e-6)

    patience, patience_counter = 10, 0
    best_loss = np.inf
    best_f1_macro = 0.0
    best_f1_weighted = 0.0

    # scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    for epoch in range(100):
        model.train()
        train_loss, total_correct, total_samples = 0.0, 0, 0
        all_train_preds, all_train_labels = [], []

        t0 = time.time()
        for image, habitat, substrate, geo, eventDate, label, _ in tqdm(train_loader):
            image = image.to(device)
            habitat, substrate = habitat.to(device), substrate.to(device)
            geo, eventDate = geo.to(device), eventDate.to(device)
            label = label.to(device)

            # skip unlabeled rows just in case (-1)
            mask = label >= 0
            if mask.sum() == 0:
                continue
            image, habitat, substrate, geo, eventDate, label = image[mask], habitat[mask], substrate[mask], geo[mask], eventDate[mask], label[mask]

            optimizer.zero_grad(set_to_none=True)
            # with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                logits, img_tokens, meta_tokens = model(image, habitat, substrate, geo, eventDate, return_tokens=True)
                ce_loss = ce(logits, label)
                gw_loss = gaussian_wasserstein_loss(img_tokens, meta_tokens)
                loss = ALPHA_CE * ce_loss + BETA_GW * gw_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            preds = logits.argmax(1).detach().cpu().numpy()
            labels_np = label.detach().cpu().numpy()
            all_train_preds.append(preds)
            all_train_labels.append(labels_np)
            total_correct += (preds == labels_np).sum()
            total_samples += labels_np.size

        # train metrics
        all_train_preds = np.concatenate(all_train_preds) if all_train_preds else np.array([])
        all_train_labels = np.concatenate(all_train_labels) if all_train_labels else np.array([])
        train_acc = (total_correct / total_samples) if total_samples else 0.0
        train_f1_macro = f1_score(all_train_labels, all_train_preds, average='macro', zero_division=0) if total_samples else 0.0
        train_f1_weighted = f1_score(all_train_labels, all_train_preds, average='weighted', zero_division=0) if total_samples else 0.0
        avg_train_loss = train_loss / max(1, len(train_loader))

        # validation
        model.eval()
        val_loss, v_correct, v_samples = 0.0, 0, 0
        all_val_preds, all_val_labels = [], []
        with torch.no_grad():
            for image, habitat, substrate, geo, eventDate, label, _ in tqdm(valid_loader):
                image = image.to(device)
                habitat, substrate = habitat.to(device), substrate.to(device)
                geo, eventDate = geo.to(device), eventDate.to(device)
                label = label.to(device)

                mask = label >= 0
                if mask.sum() == 0: continue
                image, habitat, substrate, geo, eventDate, label = image[mask], habitat[mask], substrate[mask], geo[mask], eventDate[mask], label[mask]

                logits, img_tokens, meta_tokens = model(image, habitat, substrate, geo, eventDate, return_tokens=True)
                ce_loss = ce(logits, label)
                gw_loss = gaussian_wasserstein_loss(img_tokens, meta_tokens)
                loss = ALPHA_CE * ce_loss + BETA_GW * gw_loss

                val_loss += loss.item()
                preds = logits.argmax(1).detach().cpu().numpy()
                labels_np = label.detach().cpu().numpy()
                all_val_preds.append(preds)
                all_val_labels.append(labels_np)
                v_correct += (preds == labels_np).sum()
                v_samples += labels_np.size

        all_val_preds = np.concatenate(all_val_preds) if all_val_preds else np.array([])
        all_val_labels = np.concatenate(all_val_labels) if all_val_labels else np.array([])
        val_acc = (v_correct / v_samples) if v_samples else 0.0
        val_f1_macro = f1_score(all_val_labels, all_val_preds, average='macro', zero_division=0) if v_samples else 0.0
        val_f1_weighted = f1_score(all_val_labels, all_val_preds, average='weighted', zero_division=0) if v_samples else 0.0
        avg_val_loss = val_loss / max(1, len(valid_loader))

        lr_now = optimizer.param_groups[0]['lr']
        dt = time.time() - t0
        print(f"Epoch {epoch+1:03d} | "
              f"train loss {avg_train_loss:.4f} acc {train_acc:.4f} f1M {train_f1_macro:.4f} f1W {train_f1_weighted:.4f} | "
              f"val loss {avg_val_loss:.4f} acc {val_acc:.4f} f1M {val_f1_macro:.4f} f1W {val_f1_weighted:.4f} | "
              f"{dt:.1f}s  lr {lr_now:.2e}")

        log_epoch_to_csv(csv_file_path, [
            epoch+1, f"{dt:.2f}",
            f"{avg_train_loss:.6f}", f"{train_acc:.6f}", f"{train_f1_macro:.6f}", f"{train_f1_weighted:.6f}",
            f"{avg_val_loss:.6f}", f"{val_acc:.6f}", f"{val_f1_macro:.6f}", f"{val_f1_weighted:.6f}",
            f"{lr_now:.6f}"
        ])

        # checkpoint on F1s and loss
        updated_ckpt = False
        if val_f1_macro > best_f1_macro:
            best_f1_macro = val_f1_macro
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_f1_macro.pth"))
            updated_ckpt = True

        if val_f1_weighted > best_f1_weighted:
            best_f1_weighted = val_f1_weighted
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_f1_weighted.pth"))
            updated_ckpt = True

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_loss.pth"))
            updated_ckpt = True
            patience_counter = 0
        else:
            patience_counter += 1

        scheduler.step(avg_val_loss)

        if not updated_ckpt:
            # no progress this epoch on any target metric
            pass

        if patience_counter >= patience:
            print(f"Early stopping: no val loss improvement for {patience} epochs.")
            break

def evaluate_network_on_test_set(data_file, image_path, checkpoint_dir, session_name):
    ensure_folder(checkpoint_dir)
    output_csv_path = os.path.join(checkpoint_dir, "test_predictions.csv")

    # Prefer best F1 Macro, fall back to best weighted, else best loss
    p_macro = os.path.join(checkpoint_dir, "best_f1_macro.pth")
    p_weight = os.path.join(checkpoint_dir, "best_f1_weighted.pth")
    p_loss = os.path.join(checkpoint_dir, "best_loss.pth")
    if   os.path.exists(p_macro): best_trained_model = p_macro
    elif os.path.exists(p_weight): best_trained_model = p_weight
    else: best_trained_model = p_loss

    df = pd.read_csv(data_file)
    test_df = df[df['filename_index'].str.startswith('fungi_test')].copy().sort_index()
    test_dataset = FungiDataset(test_df, image_path, transform=get_transforms('valid'))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CMSEMixNet(num_classes=NUM_CLASSES, dims=test_dataset.dims).to(device)
    model.load_state_dict(torch.load(best_trained_model, map_location=device))
    model.eval()

    results = []
    with torch.no_grad():
        for image, habitat, substrate, geo, eventDate, _, filename in tqdm(test_loader, desc="Evaluating"):
            image = image.to(device)
            habitat, substrate = habitat.to(device), substrate.to(device)
            geo, eventDate = geo.to(device), eventDate.to(device)
            logits = model(image, habitat, substrate, geo, eventDate)
            preds = logits.argmax(1).cpu().numpy()
            results.extend(zip(filename, preds))

    with open(output_csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([session_name])
        writer.writerows(results)
    print(f"Results saved to {output_csv_path}")

# ------------------------------
# Main
# ------------------------------

if __name__ == "__main__":
    seed_torch(777)
    image_path = 'data/'
    data_file = 'data/metadata.csv'
    session = "Experiment0"
    checkpoint_dir = os.path.join(f"results/{session}/")

    train_fungi_network(data_file, image_path, checkpoint_dir)
    evaluate_network_on_test_set(data_file, image_path, checkpoint_dir, session)
