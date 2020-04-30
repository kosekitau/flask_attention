#!/usr/bin/env python
# coding: utf-8

# In[1]:


# パッケージのimport
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
#埋め込み層の定義

class Embedder(nn.Module):
  def __init__(self, text_embedding_vectors):
    super(Embedder, self).__init__()
    #freezeで更新をしない
    self.embeddings=nn.Embedding.from_pretrained(
        embeddings=text_embedding_vectors, freeze=True)

  def forward(self, x):
    x_vec = self.embeddings(x)

    return x_vec

import math
class PositionalEncoder(nn.Module):
    '''入力された単語の位置を示すベクトル情報を付加する'''

    def __init__(self, d_model=300, max_seq_len=256):
        super().__init__()

        self.d_model = d_model  # 単語ベクトルの次元数

        # 単語の順番（pos）と埋め込みベクトルの次元の位置（i）によって一意に定まる値の表をpeとして作成
        pe = torch.zeros(max_seq_len, d_model)

        # GPUが使える場合はGPUへ送る、実際に学習時には使用する
        #学習時以外(動作確認)で使うとerrorでるっぽい？
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #pe = pe.to(device)

        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos /
                                          (10000 ** ((2 * (i + 1))/d_model)))

        # 表peの先頭に、ミニバッチ次元となる次元を足す
        self.pe = pe.unsqueeze(0)

        # 勾配を計算しないようにする
        self.pe.requires_grad = False

    def forward(self, x):

        # 入力xとPositonal Encodingを足し算する
        # xがpeよりも小さいので、大きくする
        ret = math.sqrt(self.d_model)*x + self.pe
        return ret


class Attention(nn.Module):
  def __init__(self, d_model=300):
    super().__init__()

    #query, value, keyを出力
    self.q_linear = nn.Linear(d_model, d_model)
    self.v_linear = nn.Linear(d_model, d_model)
    self.k_linear = nn.Linear(d_model, d_model)

    #ここは最後のoutput
    self.out = nn.Linear(d_model, d_model)

    self.d_k = d_model

  def forward(self, q, k, v, mask):
    #query, value, keyを出力
    k = self.k_linear(k)
    q = self.q_linear(q)
    v = self.v_linear(v)

    #queryに対するmemoryの関連度を計算している, 大きさの制限もかけている
    weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.d_k)

    mask = mask.unsqueeze(1)
    #mask部分を-infで置き換え
    weights = weights.masked_fill(mask==0, -1e9)

    """
    softmaxで関連度を正規化
    ここはshapeが(queryの単語数,　memoryの単語数)になっており
    各行はqueryに対して各memoryの関連度(のようなもの)を表している
    これを関数から出力することで可視化に使っている。
    """
    normlized_weights = F.softmax(weights, dim=-1)

    #Attention
    output = torch.matmul(normlized_weights, v)
    output = self.out(output)

    return output, normlized_weights


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=1024, dropout=0.1):
        '''Attention層から出力を単純に全結合層2つで特徴量を変換するだけのユニットです'''
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.dropout(F.relu(x))
        x = self.linear_2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()

        # LayerNormalization層
        # https://pytorch.org/docs/stable/nn.html?highlight=layernorm
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

        # Attention層
        self.attn = Attention(d_model)

        # Attentionのあとの全結合層2つ
        self.ff = FeedForward(d_model)

        # Dropout
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 正規化とAttention
        x_normlized = self.norm_1(x)
        output, normlized_weights = self.attn(
            x_normlized, x_normlized, x_normlized, mask)
        
        x2 = x + self.dropout_1(output)

        # 正規化と全結合層
        x_normlized2 = self.norm_2(x2)
        output = x2 + self.dropout_2(self.ff(x_normlized2))

        return output, normlized_weights


  
class ClassificationHead(nn.Module):
  def __init__(self, d_model=300, output_dim=5):
    super().__init__()

    self.linear = nn.Linear(d_model, output_dim)
    
    nn.init.normal_(self.linear.weight, std=0.02)
    nn.init.normal_(self.linear.bias, 0)

  def forward(self, x):
    #各バッチの<cls>の特徴量を抽出する
    x0 = x[:, 0, :]
    out = self.linear(x0)

    return out


class TransformerClassification(nn.Module):
  def __init__(self, text_embedding_vectors, d_model=300, max_seq_len=140,
               output_dim=5):
    super().__init__()

    self.net1 = Embedder(text_embedding_vectors)
    self.net2 = PositionalEncoder(d_model=d_model, max_seq_len=max_seq_len)
    self.net3_1 = TransformerBlock(d_model=d_model)
    self.net3_2 = TransformerBlock(d_model=d_model)
    self.net4 = ClassificationHead(output_dim=output_dim, d_model=d_model)

  def forward(self, x, mask):
    x1 = self.net1(x)
    x2 = self.net2(x1)
    x3_1, normlized_weights_1 = self.net3_1(x2, mask)
    x3_2, normlized_weights_2 = self.net3_2(x3_1, mask)
    x4 = self.net4(x3_2,)
    return x4, normlized_weights_1, normlized_weights_2


# In[2]:


import pickle
#重み
x = np.load('omomi.npy')
x = torch.from_numpy(x.astype(np.float32)).clone()

#辞書
itos = pickle.load(open('itos.pkl', 'rb'))
stoi = pickle.load(open('stoi.pkl', 'rb'))

model_path = 'net.pth'
model = TransformerClassification(
    text_embedding_vectors=x, d_model=300, max_seq_len=140, output_dim=5)
model.load_state_dict(torch.load(model_path))


# In[3]:


def text_to_ids(text_list, vcb):
  result = torch.zeros(140, dtype=torch.long)
  result[0] = vcb['<cls>']
  for i, word in enumerate(text_list):
    if word in vcb:
      result[i+1] = vcb[word]
    else:
      result[i+1] = vcb['<unk>']
  for j in range(i+1, 139):
    result[j+1] = vcb['<pad>']

  return result


# In[4]:


from IPython.display import HTML
# HTMLを作成する関数を実装


def highlight(word, attn):
    "Attentionの値が大きいと文字の背景が濃い赤になるhtmlを出力させる関数"

    html_color = '#%02X%02X%02X' % (
        255, int(255*(1 - attn)), int(255*(1 - attn)))
    return '<span style="background-color: {}"> {}</span>'.format(html_color, word)


def mk_html(index, text, preds, normlized_weights_1, normlized_weights_2, vcb):
    "HTMLデータを作成する"

    # indexの結果を抽出
    sentence = text  # 文章
    #label = batch.Label[index]  # ラベル
    pred = preds  # 予測

    # indexのAttentionを抽出と規格化
    #index番目のデータの0番目のmemoryの関連度を抽出している
    html = '可視化ワード：{}<br><br>'.format(vcb[text[index]])
    attens1 = normlized_weights_1[0, index, :]  # 0番目の<cls>のAttention
    attens1 /= attens1.max()

    attens2 = normlized_weights_2[0, index, :]  # 0番目の<cls>のAttention
    attens2 /= attens2.max()

    # ラベルと予測結果を文字に置き換え
    #label_str = label
    pred_str = pred

    # 表示用のHTMLを作成する
    html += '推論ラベル：{}<br><br>'.format(pred_str)

    # 1段目のAttention
    html += '[TransformerBlockの1段目のAttentionを可視化]<br>'
    for word, attn in zip(sentence, attens1):
        html += highlight(vcb[word], attn)
    html += "<br><br>"

    # 2段目のAttention
    html += '[TransformerBlockの2段目のAttentionを可視化]<br>'
    for word, attn in zip(sentence, attens2):
        html += highlight(vcb[word], attn)

    html += "<br><br>"

    return html


# In[5]:


# Transformerで処理
import torch.nn.functional as F 
import MeCab


mecab = MeCab.Tagger('-Owakati')

# バッチサイズ分文章とラベルのセットを取り出す
s = 'その棚にある赤いりんごはとてもまずい'
result = [tok for tok in mecab.parse('こんにちは今日も').split()]
print(result)
result = text_to_ids(result, stoi)

inputs = torch.tensor(result)  # 文章をid列にしたもの

# mask作成
input_pad = 1  # 単語のIDにおいて、'<pad>': 1 なので
input_mask = (inputs != input_pad)

# Transformerに入力
outputs, normlized_weights_1, normlized_weights_2 = model(
    inputs, input_mask)
_, preds = torch.max(outputs, 1)  # ラベルを予測

index = 2
html_output = mk_html(index, result, preds, normlized_weights_1,
                      normlized_weights_2, itos)  # HTML作成
HTML(html_output)  # HTML形式で出力


# In[ ]:




