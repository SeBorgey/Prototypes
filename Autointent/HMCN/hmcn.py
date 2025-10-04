import torch
import torch.nn as nn

class HMCNF(nn.Module):
    def __init__(self, input_size, num_classes_per_level, hidden_size=384, dropout_rate=0.5):
        super().__init__()
        
        self.num_levels = len(num_classes_per_level)

        self.global_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.local_transition_layers = nn.ModuleList()
        self.local_output_layers = nn.ModuleList()

        current_input_size = input_size
        for i in range(self.num_levels):
            self.global_layers.append(nn.Linear(current_input_size, hidden_size))
            self.batch_norms.append(nn.BatchNorm1d(hidden_size))
            current_input_size = hidden_size + input_size
            
            self.local_transition_layers.append(nn.Linear(hidden_size, hidden_size))
            self.local_output_layers.append(nn.Linear(hidden_size, num_classes_per_level[i]))
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        local_logits_list = []
        global_h = None

        for i in range(self.num_levels):
            input_to_global = x if i == 0 else torch.cat([global_h, x], dim=1)

            global_h = self.global_layers[i](input_to_global)
            global_h = self.batch_norms[i](global_h)
            global_h = self.relu(global_h)
            global_h = self.dropout(global_h)

            local_h = self.local_transition_layers[i](global_h)
            local_h = self.relu(local_h)
            local_logits = self.local_output_layers[i](local_h)
            local_logits_list.append(local_logits)

        return local_logits_list