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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torchvision import models
from sklearn.model_selection import train_test_split
from logging import getLogger, DEBUG, FileHandler, Formatter, StreamHandler
from tqdm import tqdm
import numpy as np
from PIL import Image
import time
import csv
from collections import Counter
import wandb

from utils import get_subset

def ensure_folder(folder):
    """
    Ensure a folder exists; if not, create it.
    """
    if not os.path.exists(folder):
        print(f"Folder '{folder}' does not exist. Creating...")
        os.makedirs(folder)

def seed_torch(seed=777):
    """
    Set seed for reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def initialize_csv_logger(file_path):
    """Initialize the CSV file with header."""
    header = ["epoch", "time", "val_loss", "val_accuracy", "train_loss", "train_accuracy"]
    with open(file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)

def log_epoch_to_csv(file_path, epoch, epoch_time, train_loss, train_accuracy, val_loss, val_accuracy):
    """Log epoch summary to the CSV file."""
    with open(file_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([epoch, epoch_time, val_loss, val_accuracy, train_loss, train_accuracy])

def get_transforms(data):
    """
    Return augmentation transforms for the specified mode ('train' or 'valid').
    """
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
        raise ValueError("Unknown data mode requested (only 'train' or 'valid' allowed).")

OUT_DIR = 'processed_metadata'

def preprocess_metadata(df):
    os.makedirs(OUT_DIR, exist_ok=True)

    filled_the_gaps_path = os.path.join(OUT_DIR, 'filled_gaps.csv')
    if os.path.exists(filled_the_gaps_path):
        df = pd.read_csv(filled_the_gaps_path)
    else:
        df = fill_the_gaps(df)
        df.to_csv(filled_the_gaps_path, index=False)

    train_df = get_subset(df, 'train')

    one_hot_columns = ['Habitat', 'Substrate']
    for col in one_hot_columns:
        col_data = df[col].astype('category').cat.codes
        col_data = torch.tensor(col_data.values).long()

        col_data[col_data == -1] = col_data.max() + 1 # Handle missing values

        col_data = torch.nn.functional.one_hot(col_data, num_classes=len(df[col].unique()))
        col_data = col_data.float()[:, :-1]

        torch.save(col_data, os.path.join(OUT_DIR, f'{col}.pt'))

    geo_columns = ['Latitude', 'Longitude']
    geo_data = []
    for col in geo_columns:
        min_col = train_df[col].min()
        max_col = train_df[col].max()

        col_data = df[col]

        col_data = (((col_data - min_col) / (max_col - min_col)) - 0.5) * 2
        col_data = col_data.fillna(0.)
        col_data = torch.tensor(col_data.values, dtype=torch.float)
        geo_data.append(col_data)

    geo_data = torch.stack(geo_data, dim=1)
    torch.save(geo_data, os.path.join(OUT_DIR, 'geo.pt'))
    event_date = pd.to_datetime(df['eventDate'])

    month_sin = torch.tensor(np.sin(2 * np.pi * event_date.dt.month / 12).fillna(0.)).float()
    month_cos = torch.tensor(np.cos(2 * np.pi * event_date.dt.month / 12).fillna(0.)).float()
    day_of_year_sin = torch.tensor(np.sin(2 * np.pi * event_date.dt.dayofyear / 365.25).fillna(0.)).float()
    day_of_year_cos = torch.tensor(np.cos(2 * np.pi * event_date.dt.dayofyear / 365.25).fillna(0.)).float()
    week_sin = torch.tensor(np.sin(2 * np.pi * event_date.dt.isocalendar()['week'] / 52).fillna(0.)).float()
    week_cos = torch.tensor(np.cos(2 * np.pi * event_date.dt.isocalendar()['week'] / 52).fillna(0.)).float()

    max_year = 2020
    min_year = 1985
    year = torch.tensor(2 * (event_date.dt.year - min_year) / (max_year - min_year) - 1).float()
    year = torch.nan_to_num(year, 0.)

    time_data = torch.stack([month_sin, month_cos, day_of_year_sin, day_of_year_cos, week_sin, week_cos, year], dim=1)
    torch.save(time_data, os.path.join(OUT_DIR, 'eventDate.pt'))

def load_metadata():
    cols = ['Habitat', 'Substrate', 'geo', 'eventDate']

    data = []
    for col in cols:
        if not os.path.exists(os.path.join(OUT_DIR, f'{col}.pt')):
            raise Exception("Metadata not preprocessed!")
        data.append(torch.load(os.path.join(OUT_DIR, f'{col}.pt')))

    return data

class FungiDataset(Dataset):
    def __init__(self, df, path, transform=None):
        self.df = df
        self.transform = transform
        self.path = path

        self.habitat, self.substrate, self.geo, self.eventDate = load_metadata()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # ----------------------------------------
        # Get image part
        file_path = self.df['filename_index'].values[idx]
        # Get label if it exists; otherwise return None
        label = self.df['taxonID_index'].values[idx]  # Get label
        if pd.isnull(label):
            label = -1  # Handle missing labels for the test dataset
        else:
            label = int(label)

        with Image.open(os.path.join(self.path, file_path)) as img:
            # Convert to RGB mode (handles grayscale images as well)
            image = img.convert('RGB')
        image = np.array(image)

        # Apply transformations if available
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        # ----------------------------------------
        # Get metadata part
        habitat = self.habitat[idx]
        substrate = self.substrate[idx]
        geo = self.geo[idx]
        eventDate = self.eventDate[idx]

        return image, habitat, substrate, geo, eventDate, label, file_path

class EmbeddingNet(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.1):
        super(EmbeddingNet, self).__init__()

        self.module = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, out_features),
            nn.GELU(),
        )

    def forward(self, x):
        return self.module(x)

class BallsNetMetadata(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Base efficientnet
        self.efficient_net = models.efficientnet_b0(pretrained=True)

        out_embds = [32, 128, 128, 8]

        # Metadata processing heads
        self.temporal_embedding = EmbeddingNet(7, out_embds[0])
        self.habitat_embedding =  EmbeddingNet(29, out_embds[1])
        self.substrate_embedding = EmbeddingNet(22, out_embds[2])
        self.geo_embedding = EmbeddingNet(2, out_embds[3])

        self.classification_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.efficient_net.classifier[1].in_features + sum(out_embds), 183)  # Number of classes
        )
        # Remove the original classifier head
        self.efficient_net.classifier = nn.Identity()

    def forward(self, image, habitat, substrate, geo, eventDate):
        image = self.efficient_net(image)

        temp = self.temporal_embedding(eventDate)
        hab = self.habitat_embedding(habitat)
        subs = self.substrate_embedding(substrate)
        geo = self.geo_embedding(geo)
        x = torch.cat([image, temp, hab, subs, geo], dim=1)
        return self.classification_head(x)

BATCH_SIZE = 64
def train_fungi_network(data_file, image_path, checkpoint_dir):
    """
    Train the network and save the best models based on validation accuracy and loss.
    Incorporates early stopping with a patience of 10 epochs.
    """
    # Ensure checkpoint directory exists
    ensure_folder(checkpoint_dir)

    # Set Logger
    csv_file_path = os.path.join(checkpoint_dir, 'train.csv')
    initialize_csv_logger(csv_file_path)

    # Load metadata
    df = pd.read_csv(data_file)
    preprocess_metadata(df)

    train_df = df[df['filename_index'].str.startswith('fungi_train')]

    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['taxonID_index'], shuffle=True)
    print('Training size', len(train_df))
    print('Validation size', len(val_df))

    # Initialize DataLoaders
    train_dataset = FungiDataset(train_df, image_path, transform=get_transforms(data='train'))
    valid_dataset = FungiDataset(val_df, image_path, transform=get_transforms(data='valid'))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Network Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BallsNetMetadata()
    model = model.to(device)

    # Define Optimization, Scheduler, and Criterion
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=1, eps=1e-6)
    criterion = nn.CrossEntropyLoss()
    wandb.watch(model, criterion, log="all", log_freq=50, log_graph=True)

    # Early stopping setup
    patience = 10
    patience_counter = 0
    best_loss = np.inf
    best_accuracy = 0.0

    # Training Loop
    for epoch in range(100):  # Maximum epochs
        model.train()
        train_loss = 0.0
        total_correct_train = 0
        total_train_samples = 0
        
        # Start epoch timer
        epoch_start_time = time.time()
        
        # Training Loop
        for image, habitat, substrate, geo, eventDate, label, _ in tqdm(train_loader):
            image, habitat, substrate, geo, eventDate, label = image.to(device), habitat.to(device), substrate.to(device), geo.to(device), eventDate.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(image, habitat, substrate, geo, eventDate)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate train accuracy
            total_correct_train += (outputs.argmax(1) == label).sum().item()
            total_train_samples += label.size(0)
        
        # Calculate overall train accuracy and average loss
        train_accuracy = total_correct_train / total_train_samples
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        total_correct_val = 0
        total_val_samples = 0
        
        # Validation Loop
        with torch.no_grad():
            for image, habitat, substrate, geo, eventDate, label, _ in tqdm(valid_loader):
                image, habitat, substrate, geo, eventDate, label = image.to(device), habitat.to(device), substrate.to(device), geo.to(device), eventDate.to(device), label.to(device)
                outputs = model(image, habitat, substrate, geo, eventDate)
                val_loss += criterion(outputs, label).item()
                
                # Calculate validation accuracy
                total_correct_val += (outputs.argmax(1) == label).sum().item()
                total_val_samples += label.size(0)

        # Calculate overall validation accuracy and average loss
        val_accuracy = total_correct_val / total_val_samples
        avg_val_loss = val_loss / len(valid_loader)

        # Stop epoch timer and calculate elapsed time
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time

        # Print summary at the end of the epoch
        print(f"Epoch {epoch + 1} Summary: "
            f"Train Loss = {avg_train_loss:.4f}, Train Accuracy = {train_accuracy:.4f}, "
            f"Val Loss = {avg_val_loss:.4f}, Val Accuracy = {val_accuracy:.4f}, "
            f"Epoch Time = {epoch_time:.2f} seconds")
        
        # Log epoch metrics to the CSV file
        log_epoch_to_csv(csv_file_path, epoch + 1, epoch_time, avg_train_loss, train_accuracy, avg_val_loss, val_accuracy)
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": avg_train_loss,
            "train/acc":  train_accuracy,
            "val/loss":   avg_val_loss,
            "val/acc":    val_accuracy,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_sec": epoch_time,
        })

        # Save Models Based on Accuracy and Loss
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_accuracy.pth"))
            print(f"Epoch {epoch + 1}: Best accuracy updated to {best_accuracy:.4f}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_loss.pth"))
            print(f"Epoch {epoch + 1}: Best loss updated to {best_loss:.4f}")
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1

        # Early stopping condition
        if patience_counter >= patience:
            print(f"Early stopping triggered. No improvement in validation loss for {patience} epochs.")
            break

def evaluate_network_on_test_set(data_file, image_path, checkpoint_dir, session_name):
    """
    Evaluate network on the test set and save predictions to a CSV file.
    """
    # Ensure checkpoint directory exists
    ensure_folder(checkpoint_dir)

    # Model and Test Setup
    best_trained_model = os.path.join(checkpoint_dir, "best_accuracy.pth")
    output_csv_path = os.path.join(checkpoint_dir, "test_predictions.csv")

    df = pd.read_csv(data_file)
    test_df = df[df['filename_index'].str.startswith('fungi_test')]
    test_dataset = FungiDataset(test_df, image_path, transform=get_transforms(data='valid'))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BallsNetMetadata()
    model.load_state_dict(torch.load(best_trained_model))
    model.to(device)

    # Collect Predictions
    results = []
    model.eval()
    with torch.no_grad():
        for image, habitat, substrate, geo, eventDate, _, filename in tqdm(test_loader, desc="Evaluating"):
            image, habitat, substrate, geo, eventDate = image.to(device), habitat.to(device), substrate.to(device), geo.to(device), eventDate.to(device)
            outputs = model(image, habitat, substrate, geo, eventDate).argmax(1).cpu().numpy()
            results.extend(zip(filename, outputs))  # Store filenames and predictions only

    # Save Results to CSV
    with open(output_csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([session_name])  # Write session name as the first line
        writer.writerows(results)  # Write filenames and predictions
    print(f"Results saved to {output_csv_path}")

def fill_the_gaps(df):
    """
    Fill missing values in a dataframe:
    - cat_cols: filled randomly based on distribution within the same class
    - num_cols: pick another value of the same class and shift slightly
    - datetime_col: pick another datetime of same class and shift slightly
    """
    df_filled = df.copy()

    for cls, group in df.groupby('taxonID_index'):
        idxs = group.index

        # Fill categorical columns
        for col in ['Habitat', 'Substrate']:
            missing_idx = group[group[col].isna()].index
            if len(missing_idx) == 0:
                continue
            # Compute value counts and probabilities
            counts = group[col].value_counts(normalize=True)
            values = counts.index.values
            probs = counts.values
            # Sample according to distribution
            df_filled.loc[missing_idx, col] = np.random.choice(values, size=len(missing_idx), p=probs)

        # Fill coordinates together
        missing_coords_idx = group[group['Latitude'].isna() | group['Longitude'].isna()].index
        valid_coords = group.dropna(subset=['Latitude', 'Longitude'])[['Latitude', 'Longitude']].values
        if len(valid_coords) > 0 and len(missing_coords_idx) > 0:
            for mi in missing_coords_idx:
                coord = valid_coords[np.random.randint(len(valid_coords))]
                shift = np.random.uniform(-0.01, 0.01, size=2)
                df_filled.loc[mi, ['Latitude', 'Longitude']] = coord + shift

        # Fill datetime column
        if 'eventDate' in df.columns:
            missing_idx = group[group['eventDate'].isna()].index
            valid_vals = group['eventDate'].dropna().values
            if len(valid_vals) == 0:
                continue
            for mi in missing_idx:
                val = np.random.choice(valid_vals)
                # small shift in hours
                shift = int(np.random.uniform(-10, 10))
                if int(val.split('-')[-1]) + shift > 28:
                    day=28 # TODO: Hotfix for the day of the month, ideally we should be able to generate every day
                elif int(val.split('-')[-1]) + shift<1:
                    day=1
                else:
                    day = int(val.split('-')[-1]) + shift
                parts = str(val).split('-')          # split by dash
                parts[-1] = "{:02}".format(day)                      # replace last element
                df_filled.at[mi, 'eventDate'] = '-'.join(parts)  # join back and assign

    return df_filled

if __name__ == "__main__":
    # Path to fungi images
    image_path = 'data/FungiImages/'
    # Path to metadata file
    data_file = str('data/metadata.csv')

    # Session name: Change session name for every experiment! 
    # Session name will be saved as the first line of the prediction file
    session = "BallsNet_filled_gaps"

    wandb.init(
        project="fungi-metadata",           # ← change project name if you want
        name=session,
        config={
            "batch_size": BATCH_SIZE,
            "optimizer": "Adam",
            "lr": 0.001,
            "epochs": 100,
            "image_size": 224,
            "model": "BallsNetMetadata",
            "classes": 183,
            "augmentations": ["RandomResizedCrop","HFlip","VFlip","RandBrightContrast"],
        },
    )

    # Folder for results of this experiment based on session name:
    checkpoint_dir = os.path.join(f"results/{session}/")

    train_fungi_network(data_file, image_path, checkpoint_dir)
    evaluate_network_on_test_set(data_file, image_path, checkpoint_dir, session)

    wandb.finish()