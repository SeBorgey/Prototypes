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
import requests
import zipfile

from mlgcn_model import TextMLGCN

CONFIG = {
    # Data and Models
    "dataset_name": "DeepPavlov/events",
    "bert_model_name": "bert-base-uncased",
    "label_embedding_type": "glove", 
    "label_embedding_dim": 300, # For GloVe, must be 50, 100, 200, or 300
    "cache_dir": "./.cache",

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

def _ensure_glove_is_ready(config):
    dim = config["label_embedding_dim"]
    glove_filename = f"glove.6B.{dim}d.txt"
    glove_path = f"./{glove_filename}"

    if os.path.exists(glove_path):
        print(f"Glove file found: {glove_path}")
        return glove_path

    print(f"Glove file not found. Starting download for dimension {dim}...")
    
    glove_zip_url = 'http://nlp.stanford.edu/data/glove.6B.zip'
    os.makedirs(config["cache_dir"], exist_ok=True)
    zip_filename = os.path.join(config["cache_dir"], 'glove.6B.zip')
    
    try:
        response = requests.get(glove_zip_url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_filename, 'wb') as f, tqdm(
            desc="Downloading glove.6B.zip", total=total_size, unit='iB',
            unit_scale=True, unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))

        print(f"\nUnzipping required file: {glove_filename}...")
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extract(glove_filename, path='.')
        
        print(f"Glove file '{glove_filename}' extracted successfully.")
        os.remove(zip_filename)
        print("Cleaned up zip file.")
        return glove_path

    except Exception as e:
        print(f"An error occurred during Glove download/unzip: {e}")
        if os.path.exists(zip_filename): os.remove(zip_filename)
        raise

class LabelEmbeddingFactory:
    def __init__(self, num_classes, embedding_dim):
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

    def create_random(self):
        print("Using randomly initialized label embeddings.")
        return torch.randn(self.num_classes, self.embedding_dim)

    def create_one_hot(self):
        print("Using one-hot label embeddings.")
        return torch.eye(self.num_classes, self.embedding_dim)

    def create_from_glove(self, label_names, file_path):
        print(f"Loading Glove embeddings from: {file_path}")
        embeddings_index = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = vector
        
        embedding_matrix = torch.randn(self.num_classes, self.embedding_dim)
        found_count = 0
        for i, name in enumerate(label_names):
            vector = embeddings_index.get(name.split('_')[-1])
            if vector is not None:
                embedding_matrix[i] = torch.from_numpy(vector)
                found_count += 1
        
        print(f"--> Glove Status: Found vectors for {found_count} out of {self.num_classes} labels.")
        return embedding_matrix

def get_or_create_bert_embeddings(config, split_name, texts):
    os.makedirs(config["cache_dir"], exist_ok=True)
    
    safe_dataset_name = config["dataset_name"].replace('/', '_')
    safe_bert_name = config["bert_model_name"].replace('/', '_')
    cache_filename = f"emb_{split_name}_{safe_dataset_name}_{safe_bert_name}.pt"
    cache_path = os.path.join(config["cache_dir"], cache_filename)
    
    if os.path.exists(cache_path):
        print(f"Loading {split_name} BERT embeddings from cache: {cache_path}")
        # --- ИЗМЕНЕНИЕ ЗДЕСЬ ---
        return torch.load(cache_path, weights_only=True)
    else:
        print(f"Cache not found for {split_name}. Calculating BERT embeddings...")
        tokenizer = AutoTokenizer.from_pretrained(config["bert_model_name"])
        bert_model = AutoModel.from_pretrained(config["bert_model_name"])
        
        def _get_embeddings_from_texts(texts_to_process, model, tokenizer, device):
            model.to(device)
            model.eval()
            all_embeddings = []
            
            for i in tqdm(range(0, len(texts_to_process), 32), desc="Calculating BERT Embeddings"):
                batch_texts = texts_to_process[i:i + 32]
                inputs = tokenizer(
                    batch_texts, return_tensors="pt", padding=True, 
                    truncation=True, max_length=512
                ).to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                all_embeddings.append(embeddings.cpu())
            return torch.cat(all_embeddings, dim=0)

        embeddings = _get_embeddings_from_texts(texts, bert_model, tokenizer, config["device"])
        
        print(f"Saving {split_name} embeddings to cache: {cache_path}")
        torch.save(embeddings, cache_path)
        return embeddings

def prepare_data(config):
    dataset = load_dataset(config["dataset_name"])
    num_classes = len(dataset['train'][0]['label'])
    label_names = [f"class_{i}" for i in range(num_classes)]
    
    train_texts = [item['utterance'] for item in dataset['train']]
    test_texts = [item['utterance'] for item in dataset['test']]
    
    train_embeddings = get_or_create_bert_embeddings(config, 'train', train_texts)
    test_embeddings = get_or_create_bert_embeddings(config, 'test', test_texts)
    
    train_labels = torch.tensor([item['label'] for item in dataset['train']], dtype=torch.float)
    test_labels = torch.tensor([item['label'] for item in dataset['test']], dtype=torch.float)
    
    train_dataset = TensorDataset(train_embeddings, train_labels)
    test_dataset = TensorDataset(test_embeddings, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
    return train_loader, test_loader, train_labels, num_classes, label_names, train_embeddings.shape[1]

def train_and_evaluate(config):
    print(f"Using device: {config['device']}")
    
    glove_path = None
    if config["label_embedding_type"] == "glove":
        if config["label_embedding_dim"] not in [50, 100, 200, 300]:
            raise ValueError("For glove.6B, label_embedding_dim must be 50, 100, 200, or 300.")
        glove_path = _ensure_glove_is_ready(config)

    (
        train_loader, test_loader, full_train_labels, num_classes, 
        label_names, bert_feature_dim
    ) = prepare_data(config)

    emb_factory = LabelEmbeddingFactory(num_classes, config["label_embedding_dim"])
    if config["label_embedding_type"] == "glove":
        label_embeddings = emb_factory.create_from_glove(label_names, glove_path)
    elif config["label_embedding_type"] == "one_hot":
        config["label_embedding_dim"] = num_classes
        label_embeddings = emb_factory.create_one_hot()
    else: # random
        label_embeddings = emb_factory.create_random()

    model = TextMLGCN(
        num_classes=num_classes, bert_feature_dim=bert_feature_dim,
        label_embedding_dim=config["label_embedding_dim"],
        gcn_hidden_dims=config["gcn_hidden_dims"], p_reweight=config["p_reweight"],
        tau_threshold=config["tau_threshold"]
    )
    
    model.set_correlation_matrix(full_train_labels)
    model.to(config["device"])
    label_embeddings = label_embeddings.to(config["device"])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    loss_function = nn.BCEWithLogitsLoss()
    
    print("\nStarting training...")
    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}", leave=False)
        for bert_features, labels in progress_bar:
            bert_features, labels = bert_features.to(config["device"]), labels.to(config["device"])
            
            optimizer.zero_grad()
            logits = model(bert_features, label_embeddings)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        
        print(f"Epoch {epoch+1}, Average Loss: {total_loss / len(train_loader):.4f}")
    
    print("\nStarting evaluation...")
    model.eval()
    all_preds, all_true = [], []
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