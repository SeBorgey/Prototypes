import torch
import torch.nn as nn
import numpy as np

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, adj_matrix, features):
        support = self.linear(features)
        output = torch.matmul(adj_matrix, support)
        return output

class TextMLGCN(nn.Module):
    def __init__(
        self,
        num_classes,
        bert_feature_dim,
        label_embedding_dim,
        gcn_hidden_dims,
        p_reweight,
        tau_threshold
    ):
        super().__init__()
        self.num_classes = num_classes
        self.p_reweight = p_reweight
        self.tau_threshold = tau_threshold

        gcn_layers = []
        activation_layers = []
        
        in_dim = label_embedding_dim
        for hidden_dim in gcn_hidden_dims:
            gcn_layers.append(GCNLayer(in_dim, hidden_dim))
            activation_layers.append(nn.LeakyReLU(0.2))
            in_dim = hidden_dim
        
        gcn_layers.append(GCNLayer(in_dim, bert_feature_dim))
        activation_layers.append(nn.LeakyReLU(0.2))

        self.gcn_layers = nn.ModuleList(gcn_layers)
        self.activations = nn.ModuleList(activation_layers)
        
        self.register_buffer('correlation_matrix', torch.zeros(num_classes, num_classes))

    @staticmethod
    def create_correlation_matrix(train_labels, num_classes, p, tau):
        co_occurrence = train_labels.T @ train_labels
        num_labels_per_class = torch.diagonal(co_occurrence)

        conditional_prob = co_occurrence / num_labels_per_class.unsqueeze(1)
        conditional_prob.nan_to_num_(0)

        adj_matrix = (conditional_prob > tau).float()
        
        adj_matrix_no_self_loop = adj_matrix - torch.eye(num_classes, device=adj_matrix.device)
        sum_neighbors = adj_matrix_no_self_loop.sum(axis=1)
        
        weights_p = p / sum_neighbors
        weights_p.nan_to_num_(0)
        
        reweighted_adj = adj_matrix_no_self_loop * weights_p.unsqueeze(1)
        reweighted_adj.fill_diagonal_(1 - p)
        
        return reweighted_adj

    def set_correlation_matrix(self, train_labels):
        corr_matrix = self.create_correlation_matrix(
            train_labels,
            self.num_classes,
            self.p_reweight,
            self.tau_threshold
        )
        self.correlation_matrix.data.copy_(corr_matrix)

    def forward(self, bert_features, label_embeddings):
        classifiers = label_embeddings
        for i in range(len(self.gcn_layers)):
            classifiers = self.gcn_layers[i](self.correlation_matrix, classifiers)
            classifiers = self.activations[i](classifiers)
        
        logits = torch.matmul(bert_features, classifiers.T)
        return logits

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NUM_CLASSES = 20
    BERT_FEATURE_DIM = 768
    LABEL_EMBEDDING_DIM = 300
    GCN_HIDDEN_DIMS = [1024]
    P_REWEIGHT = 0.2
    TAU_THRESHOLD = 0.4
    
    BATCH_SIZE = 16
    NUM_TRAIN_SAMPLES = 1000

    model = TextMLGCN(
        num_classes=NUM_CLASSES,
        bert_feature_dim=BERT_FEATURE_DIM,
        label_embedding_dim=LABEL_EMBEDDING_DIM,
        gcn_hidden_dims=GCN_HIDDEN_DIMS,
        p_reweight=P_REWEIGHT,
        tau_threshold=TAU_THRESHOLD
    )

    train_labels = torch.from_numpy(
        np.random.randint(0, 2, size=(NUM_TRAIN_SAMPLES, NUM_CLASSES))
    ).float()
    
    model.set_correlation_matrix(train_labels)
    
    model.to(device)
    
    label_word_embeddings = torch.randn(NUM_CLASSES, LABEL_EMBEDDING_DIM).to(device)
    batch_bert_features = torch.randn(BATCH_SIZE, BERT_FEATURE_DIM).to(device)
    batch_labels = train_labels[:BATCH_SIZE].to(device)
    
    model.train()
    logits = model(batch_bert_features, label_word_embeddings)
    
    loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(logits, batch_labels)
    
    print("Model created successfully.")
    print(f"Device: {device}")
    print(f"Correlation matrix shape: {model.correlation_matrix.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Batch labels shape: {batch_labels.shape}")
    print(f"Calculated loss: {loss.item():.4f}")
    
    model.eval()
    with torch.no_grad():
        test_logits = model(batch_bert_features, label_word_embeddings)
        predictions = torch.sigmoid(test_logits)
    
    print(f"Sample prediction for first item in batch:\n{predictions[0].cpu().numpy().round(2)}")