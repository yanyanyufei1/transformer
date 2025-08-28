# description: Grouped-Query Attention implementation in PyTorch
import torch
import torch.nn as nn

class GroupQueryAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_groups):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = hidden_size // num_heads

        # Define linear layers for query, key, and value
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.num_groups * self.head_dim)
        self.v_proj = nn.Linear(self.hidden_size, self.num_groups * self.head_dim)

        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, hidden_states, causal_mask=None, padding_mask=None):
        batch_size , seq_len, _ = hidden_states.size()
        # (batch_size, seq_len, hidden_size) * (hidden_size, hidden_size) = (batch_size, seq_len, hidden_size)
        query_states = self.q_proj(hidden_states)
        # (batch_size, seq_len, hidden_size) * (hidden_size, num_groups * head_dim) = (batch_size, seq_len, num_groups * head_dim)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape query to (batch_size, num_heads, seq_len, head_dim)
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        # Reshape key and value to (batch_size, num_groups, seq_len, head_dim)
        key_states = key_states.view(batch_size, seq_len, self.num_groups, self.head_dim).transpose(1, 2).contiguous()
        value_states = value_states.view(batch_size, seq_len, self.num_groups, self.head_dim).transpose(1, 2).contiguous()

        # Compute attention scores
        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) / (self.head_dim ** 0.5)
        if causal_mask is not None:
            attention_scores += causal_mask * -1e9
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)
            attention_scores += padding_mask * -1e9
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, value_states)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        attention_output = self.o_proj(attention_output)
        return attention_output 
if __name__ == "__main__":
    batch_size = 4
    seq_len = 8
    hidden_size = 16
    num_heads = 2
    num_groups = 2

    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    # 生成因果掩码（三角矩阵）
    causal_mask = torch.triu(torch.ones(batch_size,seq_len), diagonal=1).bool()
    gqa = GroupQueryAttention(hidden_size, num_heads, num_groups)
    output = gqa(hidden_states)
    print(output.shape)  # Expected output shape: (4, 8, 16)


