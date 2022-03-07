import os
import random
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

import math
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from random import *

def dictionary(sentences):
    word_2_id = {'PAD': 0, 'CLS': 1, 'SEP': 2, 'MASK':3}
    id_2_word = {0: 'PAD', 1: 'CLS', 2: 'SEP', 3:'MASK'}
    shuffle(sentences)
    for i in range(len(sentences)):
        if sentences[i] not in word_2_id:
            word_2_id[sentences[i]] = len(word_2_id)
            id_2_word[len(id_2_word)] = sentences[i]
    return word_2_id, id_2_word

def make_batch(sentences,batch_size):
    input_ids = []
    segment_ids = []
    masked_token = []
    masked_position = []
    IsNext = []
    positive = negative = 0
    while positive != batch_size/2 or negative != batch_size/2:
        num_a = randint(0, len(sentences)-1)
        while True:
            num_b = randint(0, len(sentences)-1)
            if num_b != num_a:
                break
        a_and_b = 'CLS ' + ' '.join(sentences[num_a]) + ' SEP ' + ' '.join(sentences[num_b]) + ' SEP'
        token_ids = []
        for a in a_and_b.split():
            token_ids.append(word_2_id[a])
        seg_ids = [0] * (len(sentences[num_a]) + 2) + [1] * (len(sentences[num_b]) + 1)

        mask_num = min(max_mask,math.ceil(len(token_ids) * 0.15))
        mask_token, mask_position = [], []
        for i in range(mask_num):
            while True:
                num = randint(0,len(token_ids)-1)
                if token_ids[num] != 0 and token_ids[num] != 1 and token_ids[num] != 2 and token_ids[num] != 3:
                    break
            mask_token.append(token_ids[num])
            if randint(0,1)<0.8:
                token_ids[num] = 3
            elif randint(0,1)<0.5:
                while True:
                    num1 = randint(0, len(token_ids) - 1)
                    if token_ids[num1] != 0 and token_ids[num1] != 1 and token_ids[num1] != 2 and token_ids[num] != 3:
                        break
                token_ids[num] = token_ids[num1]
            mask_position.append(num)

        temp_token = seq_max_len - len(token_ids)
        token_ids = token_ids + [0] * temp_token
        seg_ids = seg_ids + [0] * temp_token
        temp_mask = max_mask - len(mask_token)
        mask_token = mask_token + [0] * temp_mask
        mask_position = mask_position + [0] * temp_mask

        if num_a == num_b + 1 and positive != batch_size/2:
            positive = positive + 1
            IsNext.append(True)
        elif num_a != num_b and negative != batch_size/2:
            negative = negative + 1
            IsNext.append(False)
        else:
            continue

        input_ids.append(token_ids)
        segment_ids.append(seg_ids)
        masked_token.append(mask_token)
        masked_position.append(mask_position)

    return torch.LongTensor(input_ids), torch.LongTensor(segment_ids), torch.LongTensor(masked_token), torch.LongTensor(masked_position), torch.LongTensor(IsNext)


def get_attn_mask(input):
    attn_mask = input.data
    attn_mask = attn_mask.eq(0).unsqueeze(1)
    attn_mask = attn_mask.expand(batch_size, seq_max_len, seq_max_len)
    return attn_mask

class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.token_embed = nn.Embedding(len_vocab, Embedding_size)
        self.position_embed = nn.Embedding(seq_max_len, Embedding_size)
        self.seg_embed = nn.Embedding(len_vocab, Embedding_size)
        self.norm = nn.LayerNorm(Embedding_size)
    def forward(self, input_ids, segment_ids):
        position = np.arange(0, len(input_ids[0]))
        position = torch.LongTensor(position)
        position = position.unsqueeze(0).expand(6,30)
        token_embed = self.token_embed(input_ids)
        position_embed = self.position_embed(position)
        seg_embed = self.seg_embed(segment_ids)
        input_embed = self.norm(token_embed + position_embed + seg_embed)
        return input_embed #[6,30,768]

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
    def forward(self,s_q,s_k,s_v,attn_mask):
        score = torch.matmul(s_q,s_k.transpose(2,3)) / np.sqrt(K_size) #[6,12,30,30]
        score = score.masked_fill_(attn_mask, -1e9)
        score = nn.Softmax(dim=1)(score)
        score = torch.matmul(score,s_v)
        return score #score[1,8,6,64]
    
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.WQ = nn.Linear(Embedding_size, n_heads * Q_size)
        self.WK = nn.Linear(Embedding_size, n_heads * K_size)
        self.WV = nn.Linear(Embedding_size, n_heads * V_size)
        self.norm = nn.LayerNorm(Embedding_size)
    def forward(self,Q,K,V,attn_mask):
        temp_input = Q

        q = self.WQ(Q).unsqueeze(2).view(batch_size, -1, n_heads, Q_size).transpose(1, 2) #[6,12,30,64]
        k = self.WK(K).unsqueeze(2).view(batch_size, -1, n_heads, Q_size).transpose(1, 2) #[6,12,30,64]
        v = self.WV(V).unsqueeze(2).view(batch_size, -1, n_heads, Q_size).transpose(1, 2) #[6,12,30,64]

        attn_mask = attn_mask.unsqueeze(1).expand(batch_size, n_heads, seq_max_len, seq_max_len)
        attn = ScaledDotProductAttention()(q,k,v,attn_mask) #[6,12,30,64]
        attn = attn.transpose(1,2).squeeze().reshape(batch_size, seq_max_len, Embedding_size)

        return self.norm(attn + temp_input)

def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class PoswiseFeedDorwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedDorwardNet, self).__init__()
        self.a = nn.Linear(Embedding_size, FF_d)
        self.b = nn.Linear(FF_d, Embedding_size)
    def forward(self, attn):
        en_output = self.a(attn)
        en_output = gelu(en_output)
        en_output = self.b(en_output)
        return en_output

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.MultiHeadAttention = MultiHeadAttention()
        self.PoswiseFeedDorwardNet = PoswiseFeedDorwardNet()
    def forward(self, embed, attn_mask):
        attn = self.MultiHeadAttention(embed, embed, embed, attn_mask) #[6,30,768]
        en_output = self.PoswiseFeedDorwardNet(attn) #[6,30,768]
        return en_output

class CLS(nn.Module):
    def __init__(self):
        super(CLS, self).__init__()
        self.linear_a = nn.Linear(Embedding_size, Embedding_size)
        self.tanh = nn.Tanh()
        self.linear_b = nn.Linear(Embedding_size, 2)
    def forward(self,en_output):
        cls_output = self.linear_a(en_output[:, 0])
        cls_output = self.tanh(cls_output)
        cls_output =self.linear_b(cls_output)
        return cls_output

class MLM(nn.Module):
    def __init__(self):
        super(MLM, self).__init__()
        self.linear_a = nn.Linear(Embedding_size,Embedding_size)
        self.norm = nn.LayerNorm(Embedding_size)
        self.linear_b = nn.Linear(Embedding_size, len_vocab, bias=False)
        self.softmax = nn.Softmax(dim=2)
    def forward(self,en_output,masked_position):
        masked_position = masked_position.unsqueeze(1).expand(batch_size, Embedding_size, max_mask).transpose(1,2) #[6,5,768]
        mask = torch.gather(en_output, 1, masked_position) #[6,5,768]
        mlm_output = self.linear_a(mask)
        mlm_output = gelu(mlm_output)
        mlm_output = self.norm(mlm_output)
        mlm_output = self.linear_b(mlm_output)
        mlm_output = self.softmax(mlm_output)

        return mlm_output

        return None

class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.embed = Embedding()
        self.encoder = nn.ModuleList(Encoder() for _ in range(n_layers))
        self.cls = CLS()
        self.mlm = MLM()
    def forward(self, input_ids, segment_ids, masked_position):
        embed = self.embed(input_ids,segment_ids) #[6,30,768]
        attn_mask = get_attn_mask(input_ids) #[6,30,30]
        for encoder in self.encoder:
            en_outpput = encoder(embed, attn_mask) #[6,30,768]
        cls_output = self.cls(en_outpput) #[6,2]
        mlm_output = self.mlm(en_outpput, masked_position) #[6,5,29]
        return cls_output, mlm_output

if __name__ == '__main__':
    seq_max_len = 30
    batch_size = 6
    max_mask = 5
    n_layers = 6
    n_heads = 12
    Embedding_size = 768
    FF_d = 3072
    K_size = Q_size = V_size = 64
    max_pred = 5
    maxlen = 30

    text = (
        'Hello, I am jack, where are you from?\n'
        'Hello, jack. My name is Mary. I come from Japan.\n'
        'Japan is a nice city. I want to visit there.\n'
        'Great. Next time you go to Japan, I can be your guide.\n'
        'That is good. I wish this day coming soon.\n'
        'Me too. I can not wait.'
    )
    sentence = re.sub("[.,!?\\-]", '',text.lower()).split('\n')
    sentences = []
    for i in range(len(sentence)):
        temp = sentence[i].split()
        sentences.append(temp)
    tempo = [i for j in sentences for i in j]
    word_2_id, id_2_word = dictionary(tempo)
    len_vocab = len(word_2_id)


    sentences_ids = []
    for i in range(len(sentences)):
        temp = []
        for j in range(len(sentences[i])):
            temp.append(word_2_id[sentences[i][j]])
        sentences_ids.append(temp)

    #input_ids[6,30], segment_ids[6,30], masked_token[6,5], masked_position[6,5], IsNext[6]
    input_ids, segment_ids, masked_token, masked_position, IsNext = make_batch(sentences, batch_size)
    #print(input_ids[0],'\n',segment_ids[0],'\n',masked_token[0],'\n',masked_position[0],'\n',IsNext[0])

    model = BERT()

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr = 0.0001)

    for epoch in range(100):
        cls_output, mlm_output = model(input_ids, segment_ids, masked_position) #cls_output:[6,2],mlm_output:[6,5,29]
        loss_mlm = criterion(mlm_output.transpose(1,2), masked_token)
        loss_mlm = (loss_mlm.float()).mean()
        loss_cls = criterion(cls_output, IsNext)
        loss = loss_cls + loss_mlm
        if epoch % 10 == 0:
            print('第',epoch,'个周期，损失为：',loss.item())
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), 'model.pkl')

    model.load_state_dict(torch.load('model.pkl'))

    print(text)
    print([id_2_word[w.item()] for w in input_ids[0] if id_2_word[w.item()] != 'PAD'])

    cls_output, mlm_output = model(input_ids, segment_ids, masked_position)
    mlm_output = mlm_output.data.max(2)[1]
    for i in range(len(mlm_output)):
        true = []
        for id in masked_token[i]:
            true.append(id_2_word[id.item()])
        print('真实被mask的单词：',true)
        pred = []
        for id in mlm_output[i]:
            pred.append(id_2_word[id.item()])
        print('预测被mask的单词：', pred)
    cls_output = cls_output.data.max(1)[1]
    for i in range(len(cls_output)):
        print('是否下一句：',True if IsNext[i] else False)
        print('预测下一句：',True if cls_output[i] else False)






