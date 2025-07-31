# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report
from tqdm import tqdm
import os

from mlgcn_model import TextMLGCN

CONFIG = {
    # Data and Models
    "dataset_name": "DeepPavlov/events",
    "bert_model_name": "bert-base-uncased",
    "label_embedding_type": "random", # "random", "one_hot", or "glove"
    "glove_file_path": "./glove.6B.300d.txt", # Required if type is "glove"
    "label_embedding_dim": 300,

    # Model Hyperparameters
    "gcn_hidden_dims": [1024],
    "p_reweight": 0.2,
    "tau_threshold": 0.4,
    
    # Training Hyperparameters
    "epochs": 10,
    "batch_size": 32,
    "learning_rate": 0.01,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

class LabelEmbeddingFactory:
    def __init__(self, num_classes, embedding_dim):
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

    def create_random(self):
        return torch.randn(self.num_classes, self.embedding_dim)

    def create_one_hot(self):
        return torch.eye(self.num_classes, self.embedding_dim)

    def create_from_glove(self, label_names, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Glove file not found at: {file_path}")
            
        embeddings_index = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = vector
        
        embedding_matrix = torch.randn(self.num_classes, self.embedding_dim)
        for i, name in enumerate(label_names):
            vector = embeddings_index.get(name)
            if vector is not None:
                embedding_matrix[i] = torch.from_numpy(vector)
        
        return embedding_matrix

def get_bert_embeddings(texts, model, tokenizer, device, batch_size=32):
    model.to(device)
    model.eval()
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="BERT Embedding"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Use mean pooling of the last hidden state
            embeddings = outputs.last_hidden_state.mean(dim=1)
        all_embeddings.append(embeddings.cpu())
        
    return torch.cat(all_embeddings, dim=0)

def prepare_data(config):
    dataset = load_dataset(config["dataset_name"])
    num_classes = len(dataset['train'][0]['label'])
    
    label_names = [f"class_{i}" for i in range(num_classes)]
    
    tokenizer = AutoTokenizer.from_pretrained(config["bert_model_name"])
    bert_model = AutoModel.from_pretrained(config["bert_model_name"])
    
    train_texts = [item['utterance'] for item in dataset['train']]
    test_texts = [item['utterance'] for item in dataset['test']]
    
    train_embeddings = get_bert_embeddings(train_texts, bert_model, tokenizer, config["device"])
    test_embeddings = get_bert_embeddings(test_texts, bert_model, tokenizer, config["device"])
    
    train_labels = torch.tensor([item['label'] for item in dataset['train']], dtype=torch.float)
    test_labels = torch.tensor([item['label'] for item in dataset['test']], dtype=torch.float)
    
    train_dataset = TensorDataset(train_embeddings, train_labels)
    test_dataset = TensorDataset(test_embeddings, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
    return train_loader, test_loader, train_labels, num_classes, label_names, train_embeddings.shape[1]

def train_and_evaluate(config):
    print(f"Using device: {config['device']}")
    
    (
        train_loader, 
        test_loader, 
        full_train_labels, 
        num_classes, 
        label_names,
        bert_feature_dim
    ) = prepare_data(config)

    emb_factory = LabelEmbeddingFactory(num_classes, config["label_embedding_dim"])
    if config["label_embedding_type"] == "glove":
        if config["label_embedding_dim"] != 300: # Example check
            print("Warning: Glove embedding dim is usually 50, 100, 200, or 300.")
        label_embeddings = emb_factory.create_from_glove(label_names, config["glove_file_path"])
    elif config["label_embedding_type"] == "one_hot":
        label_embeddings = emb_factory.create_one_hot()
        config["label_embedding_dim"] = num_classes
    else:
        label_embeddings = emb_factory.create_random()

    model = TextMLGCN(
        num_classes=num_classes,
        bert_feature_dim=bert_feature_dim,
        label_embedding_dim=config["label_embedding_dim"],
        gcn_hidden_dims=config["gcn_hidden_dims"],
        p_reweight=config["p_reweight"],
        tau_threshold=config["tau_threshold"]
    )
    
    model.set_correlation_matrix(full_train_labels)
    model.to(config["device"])
    label_embeddings = label_embeddings.to(config["device"])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    loss_function = nn.BCEWithLogitsLoss()
    
    print("Starting training...")
    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        for bert_features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            bert_features, labels = bert_features.to(config["device"]), labels.to(config["device"])
            
            optimizer.zero_grad()
            logits = model(bert_features, label_embeddings)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Average Loss: {total_loss / len(train_loader):.4f}")
    
    print("\nStarting evaluation...")
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for bert_features, labels in tqdm(test_loader, desc="Evaluating"):
            bert_features = bert_features.to(config["device"])
            logits = model(bert_features, label_embeddings)
            preds = torch.sigmoid(logits) > 0.5
            
            all_preds.append(preds.cpu().numpy())
            all_true.append(labels.cpu().numpy())
            
    all_preds = np.concatenate(all_preds, axis=0)
    all_true = np.concatenate(all_true, axis=0)
    
    report = classification_report(all_true, all_preds, target_names=label_names, zero_division=0)
    print("\nClassification Report:")
    print(report)

if __name__ == "__main__":
    train_and_evaluate(CONFIG)