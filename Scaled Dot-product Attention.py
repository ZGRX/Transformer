from torch import nn
from transformers import AutoConfig
from transformers import AutoTokenizer

import torch
from math import sqrt

import torch.nn.functional as F

model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

text = "time flies like an arrow"
inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
print(inputs.input_ids)
#创建 token embedding 层
config = AutoConfig.from_pretrained(model_ckpt)
token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
print(token_emb)
#把 token ids 变成向量
inputs_embeds = token_emb(inputs.input_ids)
print(inputs_embeds.size())

#创建查询、键、值向量序列 ，并且使用点积作为相似度函数来计算注意力分数：
Q = K = V = inputs_embeds
dim_k = K.size(-1)
scores = torch.bmm(Q, K.transpose(1,2)) / sqrt(dim_k)
print(scores.size())


#应用 softmax 函数标准化注意力权重：
weights = F.softmax(scores, dim=-1)
print(weights)
print(weights.sum(dim=-1))

#将注意力权重与值序列相乘，就实现了一个简化版的注意力机制：
attn_outputs = torch.bmm(weights, V)
print(attn_outputs.shape)
