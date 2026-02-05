import torch
import torch.nn as nn
import re
from math import sqrt

# from sub import sub_concat


# from sub import hidden_size


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


class CrossAttention(nn.Module):
    def __init__(self, dim, hidden_size_llm):
        super().__init__()
        num_heads = dim // 64
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        # wq wk wv dim * dim
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.GELU(),
            nn.Linear(4*dim, dim)
        )

        modules = [nn.Linear(1024, hidden_size_llm)]
        mlp_depth = 2
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_size_llm, hidden_size_llm))
        self.mlp = nn.Sequential(*modules)
        # print(f"self.attn type: {type(self.attn)}")

    def forward(self, query, key_value):
        # query: low resolution
        # key_value: high resolution、
        # print(f"query shape: {query.shape}")
        # print(f"key_value shape: {key_value.shape}")
        query = query.squeeze(1)
        key_value = key_value.reshape(key_value.shape[0], -1, key_value.shape[-1])
        # print('query&key&value\'s shape', query.shape, key_value.shape)
        attn_output, _ = self.attn(query, key_value, value=key_value)  # bug
        x = self.norm1(query + attn_output)
        x = self.norm2(x + self.ffn(x))
        return self.mlp(x)


class CrossMlp(nn.Module):
    def __init__(self, dim, hidden_size_llm):
        super().__init__()
        modules = [nn.Linear(dim * 5, hidden_size_llm)]
        mlp_depth = 2
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_size_llm, hidden_size_llm))
        self.mlp = nn.Sequential(*modules)
        # print(f"self.attn type: {type(self.attn)}")

    def forward(self, query, key_value):
        # query: [b,1,num_tokens, dim]
        # key_value: [b,4,num_tokens, dim]
        batch_size, num_crops, num_tokens, hidden_dim = key_value.shape
        high_res_features = torch.split(key_value, 1, dim=1)
        # x = (query, *high_res_features)
        # x = torch.cat(x, dim=-1)  # [batch,1,num_tokens,hidden_size*5]
        high_res_features = list(high_res_features)
        width, height = int(sqrt(num_tokens)), int(sqrt(num_tokens))
        for i in range(len(high_res_features)):
            high_res_features[i] = high_res_features[i].reshape(batch_size, width, height, hidden_dim)
        upper = torch.concat(high_res_features[:2], dim=2)
        lower = torch.concat(high_res_features[2:], dim=2)
        merge = torch.concat((upper, lower), dim=1)
        high_res = []
        width = merge.shape[2]
        for i in range(0, width, 2):
            for j in range(0, width, 2):
                sub = merge[:, i:i + 2, j:j + 2]
                b,h,w,c = sub.shape
                sub_flat = sub.reshape(b, h * w, c)
                sub_concat = sub_flat.reshape(b, -1)
                high_res.append(sub_concat.unsqueeze(1)) # B,1,hidden_dim*4


        key = torch.cat(high_res, dim=1)
        # b, 576, hidden_dim*4
        query = query.squeeze(1)  # [batch,num_tokens,hidden_size*5]
        x = torch.cat((query, key), dim=-1)  # [batch,num_tokens,hidden_size+hidden_size*4]
        return self.mlp(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')  # default-->linear

    if projector_type == 'cross_attention':
        return CrossAttention(config.mm_hidden_size, hidden_size_llm=config.hidden_size)

    if projector_type == 'cross_mlp':
        return CrossMlp(config.mm_hidden_size, hidden_size_llm=config.hidden_size)

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
