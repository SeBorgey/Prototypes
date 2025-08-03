import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics import average_precision_score, f1_score
import os

from mlgcn_model import TextMLGCN

def get_or_create_embeddings(texts, model, dim, cache_path):
    if os.path.exists(cache_path):
        embeddings = torch.load(cache_path)
    else:
        full_embeddings = model.encode(
            texts, 
            convert_to_tensor=True, 
            normalize_embeddings=True
        )
        embeddings = full_embeddings[:, :dim]
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(embeddings, cache_path)
    return embeddings

def prepare_data(config):
    dataset = load_dataset(config["dataset_name"], cache_dir=config["cache_dir"])
    num_classes = len(dataset['train'][0]['label'])
    
    model = SentenceTransformer(
        config["embedding_model_name"], 
        cache_folder=config["cache_dir"],
        device=config["device"]
    )
    
    def get_cache_path(split_name, dim):
        safe_ds = config["dataset_name"].replace("/", "_")
        safe_model = config["embedding_model_name"].replace("/", "_")
        return os.path.join(
            config["cache_dir"], f"emb_{split_name}_{safe_ds}_{safe_model}_{dim}.pt"
        )

    train_texts = [item['utterance'] for item in dataset['train']]
    test_texts = [item['utterance'] for item in dataset['test']]
    label_names = [f"event type {i}" for i in range(num_classes)]

    train_embeddings = get_or_create_embeddings(
        train_texts, model, config["sentence_embedding_dim"], 
        get_cache_path("train", config["sentence_embedding_dim"])
    )
    test_embeddings = get_or_create_embeddings(
        test_texts, model, config["sentence_embedding_dim"],
        get_cache_path("test", config["sentence_embedding_dim"])
    )
    label_embeddings = get_or_create_embeddings(
        label_names, model, config["label_embedding_dim"],
        get_cache_path("labels", config["label_embedding_dim"])
    )

    train_labels = torch.tensor([item['label'] for item in dataset['train']], dtype=torch.float)
    test_labels = torch.tensor([item['label'] for item in dataset['test']], dtype=torch.float)
    
    train_dataset = TensorDataset(train_embeddings, train_labels)
    test_dataset = TensorDataset(test_embeddings, test_labels)
    
    return train_dataset, test_dataset, train_labels, num_classes, label_embeddings

def run_trial(config):
    device = torch.device(config["device"])
    train_dataset, test_dataset, train_labels, num_classes, label_embeds = prepare_data(config)
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])
    
    model = TextMLGCN(
        num_classes=num_classes,
        bert_feature_dim=config["sentence_embedding_dim"],
        label_embedding_dim=config["label_embedding_dim"],
        gcn_hidden_dims=config["gcn_hidden_dims"],
        p_reweight=config["p_reweight"],
        tau_threshold=config["tau_threshold"]
    ).to(device)
    
    model.set_correlation_matrix(train_labels.to(device))
    label_embeddings = label_embeds.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    loss_function = nn.BCEWithLogitsLoss()
    
    for _ in range(config["epochs"]):
        model.train()
        for features, labels in train_loader:
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