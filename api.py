#!/usr/bin/env python
# coding: utf-8

# パッケージのimport
import numpy as np
import random
import math
import pickle
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import MeCab

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


class Sonar(object):

    def __init__(self):
      emb = np.load('omomi.npy')
      emb = torch.from_numpy(emb.astype(np.float32)).clone()
      # index→sentence ex:itos[0] => '<cls>'
      self.itos = pickle.load(open('itos.pkl', 'rb')) 
      # sentence→index ex:stoi['<cls>'] => 0
      self.stoi = pickle.load(open('stoi.pkl', 'rb')) 
      model_path = 'net.pth'
      self.model = TransformerClassification(
        text_embedding_vectors=emb, d_model=300, max_seq_len=140, output_dim=5)
      self.model.load_state_dict(torch.load(model_path)) #モデルにパラメータを当てはめる
      self.model.eval()
      self.text = ''

    def make_html(self, text, index):
      "HTMLデータを作成する"
      #テキストを形態素解析
      if text != '':
        self.text = text
      mecab = MeCab.Tagger('-Owakati')
      result = [tok for tok in mecab.parse(self.text).split()]
      inputs = text_to_ids(result, self.stoi) # 番号を振る
      #inputs = torch.tensor(inputs)
      
      # mask作成
      input_pad = 1  # <pad>は1
      input_mask = (inputs != input_pad)

      
      # 感情分類
      outputs, normlized_weights_1, normlized_weights_2 = self.model(
        inputs, input_mask)
      _, preds = torch.max(outputs, 1)  # ラベルを予測

      #label = batch.Label[index]  # ラベル
      #pred = preds  # 予測

      html = '<form action="/post" method="post" class="form-inline">'
      
      
      #index番目のデータの0番目のmemoryの関連度を抽出している
      
      #可視化するワード
      attn_word = self.itos[inputs[index]]
      attn_word = re.sub('<', '&lt;', attn_word)
      attn_word = re.sub('>', '&gt;', attn_word)
      html += '可視化ワード：{}<br><br>'.format(attn_word)
      #print(self.itos[inputs[index]])
      #0バッチ目のindex番目の単語
      attens1 = normlized_weights_1[0, index, :]  # <cls>のAttention
      attens1 /= attens1.max()

      attens2 = normlized_weights_2[0, index, :]  # <cls>のAttention
      attens2 /= attens2.max()

      # ラベルと予測結果を文字に置き換え
      #label_str = label
      #pred_str = pred

      
      # 表示用のHTMLを作成する
      html += '感情分析結果：{}<br><br>'.format(preds)
      
      #プルダウンメニューを作る
      html += '<select name="sel" class="form-control"><option value="null" disabled selected>分析ワードを選択</option>'
      for i, word in enumerate(result, 1):
        html+='<option value="{}">{}</option>'.format(i, word)
      html+='</select><button type="submit" class="btn btn-default">送信する</button><br><br>'

      
      # 1段目のAttention
      html += '[{}のAttentionWeightを可視化(1段目)]<br>'.format(attn_word)
      #sentenceはid列
      for word, attn in zip(inputs, attens1):
        html += highlight(self.itos[word], attn)
      html += "<br><br>"

      # 2段目のAttention
      html += '[{}のAttentionWeightを可視化(2段目)]<br>'.format(attn_word)
      for word, attn in zip(inputs, attens2):
        html += highlight(self.itos[word], attn)

      html += '<br><br><input type="text" class="form-control" id="name" name="name" placeholder="Name"><button type="submit" class="btn btn-default">送信する</button></form>'

      f = open('templates/test.html','w')
      f.write(html)
      f.close()
      #return html
      
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


def highlight(word, attn):
  "Attentionの値が大きいと文字の背景が濃い赤になるhtmlを出力させる関数"
  html_color = '#%02X%02X%02X' % (
    255, int(255*(1 - attn)), int(255*(1 - attn)))
  return '<span style="background-color: {}"> {}</span>'.format(html_color, word)
