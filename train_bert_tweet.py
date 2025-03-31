import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import wandb
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW

# ---------------------------
# Dataset Class Definition
# ---------------------------
class SentimentDataset(Dataset):
    """
    Custom Dataset for sentiment analysis.
    Each item is a dictionary containing tokenized input IDs, attention mask, and label.
    """
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        # Tokenize the text using BERT tokenizer with padding and truncation
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),  # flatten the tensor
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ---------------------------
# Data Loading and Preprocessing
# ---------------------------
def load_data(file_path, tokenizer, max_length, test_size=0.2):
    """
    Load the CSV file and split it into training and validation sets.
    The CSV is assumed to have 6 columns:
      0: polarity (0 = negative, 2 = neutral, 4 = positive)
      1: tweet id
      2: date
      3: query (or NO_QUERY)
      4: user
      5: tweet text
    The polarity labels are mapped to contiguous values:
      0 -> 0, 2 -> 1, 4 -> 2.
    """
    df = pd.read_csv(file_path)
    # filter out empty row
    df = df.dropna(subset=['text'])
    
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    df['sentiment'] = df['sentiment'].map(label_map)
    
    texts = df['text'].tolist()
    labels = df['sentiment'].tolist()
    
    # Split into training and validation sets (with stratification)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, max_length)
    return train_dataset, val_dataset

def load_test_data(file_path, tokenizer, max_length):
    """
    Load the CSV file for testing.
    The CSV is assumed to have columns 'sentiment' and 'text'.
    """
    df = pd.read_csv(file_path)
    # filter out empty row
    df = df.dropna(subset=['text'])

    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    df['sentiment'] = df['sentiment'].map(label_map)

    texts = df['text'].tolist()
    labels = df['sentiment'].tolist()
    test_dataset = SentimentDataset(texts, labels, tokenizer, max_length)
    return test_dataset

# ---------------------------
# Training and Evaluation Functions
# ---------------------------
def train_epoch(model, data_loader, optimizer, device, scheduler):
    """
    Train the model for one epoch.
    """
    model.train()
    losses = []
    correct_predictions = 0
    
    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    accuracy = correct_predictions.double() / len(data_loader.dataset)
    return accuracy, np.mean(losses)

def eval_model(model, data_loader, device):
    """
    Evaluate the model on the validation set and compute accuracy, average loss, and weighted F1 score.
    """
    model.eval()
    losses = []
    correct_predictions = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    accuracy = correct_predictions.double() / len(data_loader.dataset)
    avg_loss = np.mean(losses)
    f1 = f1_score(y_true, y_pred, average='weighted')
    return accuracy, avg_loss, f1

# ---------------------------
# Main Function
# ---------------------------
def main(args):
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=3)
    model = model.to(device)

    # Freeze all BERTweet parameters
    for param in model.roberta.parameters():
        param.requires_grad = False
    
    # Unfreeze the last 2 layers of the BERTweet encoder
    for layer in model.roberta.encoder.layer[-2:]:
        for param in layer.parameters():
            param.requires_grad = True

    wandb.init(project=args.wandb_project, config=vars(args))
    wandb.watch(model, log="all")
    
    # Load and preprocess the data
    train_dataset, val_dataset = load_data(args.data_path, tokenizer, args.max_length, test_size=args.test_size)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(0.1 * total_steps),
                                                num_training_steps=total_steps)
    
    best_f1 = 0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_acc, train_loss = train_epoch(model, train_loader, optimizer, device, scheduler)
        print(f"Train loss: {train_loss:.4f} | Train accuracy: {train_acc:.4f}")
        
        val_acc, val_loss, val_f1 = eval_model(model, val_loader, device)
        print(f"Validation loss: {val_loss:.4f} | Validation accuracy: {val_acc:.4f} | Validation F1: {val_f1:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc.item(),
            "val_loss": val_loss,
            "val_accuracy": val_acc.item(),
            "val_f1": val_f1
        })
        
    
    # save at the end of training
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")
    
    # Evaluate on test dataset after training
    print("Evaluating on test set...")
    test_dataset = load_test_data(args.test_data_path, tokenizer, args.max_length)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    test_acc, test_loss, test_f1 = eval_model(model, test_loader, device)
    print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f} | Test F1: {test_f1:.4f}")
    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": test_acc.item(),
        "test_f1": test_f1
    })
    
    wandb.finish()
    print("Training complete.")

# ---------------------------
# Argument Parser and Script Execution
# ---------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="BERT Fine-tuning for Sentiment Analysis")
    parser.add_argument('--data_path', type=str, default='./archive/train.csv', 
                        help='Path to CSV file containing the dataset')
    parser.add_argument('--test_data_path', type=str, default='./archive/test.csv', 
                        help='Path to CSV file containing the dataset')
    parser.add_argument('--model_name', type=str, default='vinai/bertweet-base', 
                        help='Pretrained BERT model name')
    parser.add_argument('--max_length', type=int, default=128, 
                        help='Maximum sequence length for tokenization')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Batch size for training and evaluation')
    parser.add_argument('--learning_rate', type=float, default=2e-5, 
                        help='Learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=3, 
                        help='Number of training epochs')
    parser.add_argument('--test_size', type=float, default=0.2, 
                        help='Proportion of dataset to include in validation split')
    parser.add_argument('--output_dir', type=str, default='./saved_model', 
                        help='Directory to save the fine-tuned model')
    parser.add_argument('--wandb_project', type=str, default='twitter_sentiment_analysis', 
                        help='Weights & Biases project name')
    
    args = parser.parse_args()
    main(args)