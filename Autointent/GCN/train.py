import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import average_precision_score, f1_score
from tqdm import tqdm
import os
import requests
import zipfile

from mlgcn_model import TextMLGCN

def _ensure_glove_is_ready(config):
    dim = config["label_embedding_dim"]
    glove_filename = f"glove.6B.{dim}d.txt"
    glove_path = f"./{glove_filename}"
    if os.path.exists(glove_path): return glove_path

    zip_filename = os.path.join(config["cache_dir"], 'glove.6B.zip')
    if not os.path.exists(zip_filename):
        print(f"Glove file not found. Starting download for dimension {dim}...")
        glove_zip_url = 'http://nlp.stanford.edu/data/glove.6B.zip'
        os.makedirs(config["cache_dir"], exist_ok=True)
        try:
            response = requests.get(glove_zip_url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            with open(zip_filename, 'wb') as f, tqdm(
                desc="Downloading glove.6B.zip", total=total_size, unit='iB',
                unit_scale=True, unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk); bar.update(len(chunk))
        except Exception as e:
            print(f"An error occurred during Glove download: {e}")
            if os.path.exists(zip_filename): os.remove(zip_filename)
            raise

    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extract(glove_filename, path='.')
    return glove_path

class LabelEmbeddingFactory:
    def __init__(self, num_classes, embedding_dim):
        self.num_classes, self.embedding_dim = num_classes, embedding_dim

    def create_from_glove(self, label_names, file_path):
        embeddings_index = {line.split()[0]: np.asarray(line.split()[1:], dtype='float32')
                            for line in open(file_path, 'r', encoding='utf-8')}
        embedding_matrix = torch.randn(self.num_classes, self.embedding_dim)
        for i, name in enumerate(label_names):
            vector = embeddings_index.get(name.split('_')[-1])
            if vector is not None: embedding_matrix[i] = torch.from_numpy(vector)
        return embedding_matrix
    
    # Other methods like create_one_hot can be added here if needed

def get_or_create_bert_embeddings(config, split_name, texts, verbose=True):
    os.makedirs(config["cache_dir"], exist_ok=True)
    safe_ds_name = config["dataset_name"].replace('/', '_')
    safe_bert_name = config["bert_model_name"].replace('/', '_')
    cache_path = os.path.join(config["cache_dir"], f"emb_{split_name}_{safe_ds_name}_{safe_bert_name}.pt")
    
    if os.path.exists(cache_path):
        if verbose: print(f"Loading {split_name} BERT embeddings from cache.")
        return torch.load(cache_path, map_location=config["device"], weights_only=True)
    
    if verbose: print(f"Calculating {split_name} BERT embeddings...")
    tokenizer = AutoTokenizer.from_pretrained(config["bert_model_name"])
    bert_model = AutoModel.from_pretrained(config["bert_model_name"]).to(config["device"])
    bert_model.eval()
    
    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), 32), desc="BERT Embeddings", disable=not verbose):
            batch = tokenizer(texts[i:i+32], return_tensors="pt", padding=True, truncation=True, max_length=512).to(config["device"])
            embeddings = bert_model(**batch).last_hidden_state.mean(dim=1)
            all_embeddings.append(embeddings.cpu())
            
    final_embeddings = torch.cat(all_embeddings, dim=0)
    torch.save(final_embeddings, cache_path)
    return final_embeddings.to(config["device"])

def prepare_data(config, verbose=True):
    dataset = load_dataset(config["dataset_name"], cache_dir=config["cache_dir"])
    num_classes = len(dataset['train'][0]['label'])
    bert_feature_dim = AutoModel.from_pretrained(config["bert_model_name"]).config.hidden_size
    
    train_texts = [item['utterance'] for item in dataset['train']]
    test_texts = [item['utterance'] for item in dataset['test']]
    
    train_embeddings = get_or_create_bert_embeddings(config, 'train', train_texts, verbose)
    test_embeddings = get_or_create_bert_embeddings(config, 'test', test_texts, verbose)
    
    train_labels = torch.tensor([item['label'] for item in dataset['train']], dtype=torch.float)
    test_labels = torch.tensor([item['label'] for item in dataset['test']], dtype=torch.float)
    
    train_dataset = TensorDataset(train_embeddings, train_labels)
    test_dataset = TensorDataset(test_embeddings, test_labels)
    
    return train_dataset, test_dataset, train_labels, num_classes, bert_feature_dim

def run_trial(config, verbose=True):
    device = torch.device(config["device"])
    train_dataset, test_dataset, full_train_labels, num_classes, bert_dim = prepare_data(config, verbose)
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

    glove_path = _ensure_glove_is_ready(config)
    emb_factory = LabelEmbeddingFactory(num_classes, config["label_embedding_dim"])
    label_embeddings = emb_factory.create_from_glove([f"class_{i}" for i in range(num_classes)], glove_path).to(device)
    
    model = TextMLGCN(
        num_classes=num_classes, bert_feature_dim=bert_dim,
        label_embedding_dim=config["label_embedding_dim"],
        gcn_hidden_dims=config["gcn_hidden_dims"], p_reweight=config["p_reweight"],
        tau_threshold=config["tau_threshold"]
    ).to(device)
    model.set_correlation_matrix(full_train_labels.to(device))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    loss_function = nn.BCEWithLogitsLoss()
    
    for epoch in range(config["epochs"]):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=not verbose, leave=False)
        for features, labels in progress_bar:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(features, label_embeddings)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
    
    model.eval()
    all_scores, all_true = [], []
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            logits = model(features, label_embeddings)
            all_scores.append(torch.sigmoid(logits).cpu().numpy())
            all_true.append(labels.cpu().numpy())
            
    all_scores = np.concatenate(all_scores, axis=0)
    all_true = np.concatenate(all_true, axis=0)
    
    mAP = average_precision_score(all_true, all_scores, average="macro")
    of1 = f1_score(all_true, all_scores > 0.5, average="micro")
    
    return mAP, of1