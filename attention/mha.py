import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, attention_dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)

        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, hidden_states, causal_mask=None, padding_mask=None):
        bs, seq_len, hidden_size = hidden_states.size()

        query_states = self.q_proj(hidden_states) # (bs, seq_len, hidden_size)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # 将每个头的维度拆分出来，得到形状（bs, num_heads, seq_len, head_dim）
        query_states = query_states.view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        key_states = key_states.view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        value_states = value_states.view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

        # 计算注意力分数，attention_scores形状：（bs, num_heads, seq_len, seq_len
        # （bs, num_heads, seq_len, head_dim）* （bs, num_heads, head_dim, seq_len）= （bs, num_heads, seq_len, seq_len）
        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if causal_mask is not None:
            attention_scores += causal_mask * -1e9
        
        # 添加填充位置的掩码，每个句子不一样（batch_size, seq_len)
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)
            attention_scores += padding_mask *-1e9


        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 计算注意力输出， 通过注意力概率加权值
        #（bs, num_heads, seq_len, seq_len）* （bs, num_heads, seq_len, head_dim）= （bs, num_heads, seq_len, head_dim)
        output = torch.matmul(attention_weights, value_states)

        # 对多头注意力输出进行拼接，将形状调整为（bs, batch）
        output = output.transpose(1, 2).contiguous().view(bs, seq_len, self.head_dim * self.num_heads)

        output = self.out_proj(output)
        return output
if __name__ == "__main__":
    batch_size = 4
    seq_len = 8
    hidden_size = 16
    attention_dropout = 0.01
    num_heads = 2

    hidden_state = torch.randn(batch_size, seq_len, hidden_size)

    # 生成因果掩码（三角矩阵）
    causal_mask = torch.triu(torch.ones(batch_size,seq_len), diagonal=1).bool()
    mha = MultiHeadAttention(hidden_size, num_heads, attention_dropout)
    output = mha(hidden_state)
    print(output.size())