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

torch.manual_seed(12)
np.random.seed(12)

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

    def __init__(self, d_model=300, max_seq_len=140):
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


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, head_num, dropout_rate):
        super().__init__()
        """
        d_model：出力層の次元(head_bumの倍数)
        head_num：ヘッドの数
        dropout_rate
        """
        self.d_model = d_model
        self.head_num = head_num
        self.dropout_rate = dropout_rate
    
        #特徴量変換
        self.q_linear = nn.Linear(d_model, d_model) 
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        #出力の全結合層
        self.out = nn.Linear(d_model, d_model)
        self.attention_dropout_layer = nn.Dropout(dropout_rate)   
    
    def forward(self, q, k, v, mask):
        #key, query, valueを生成
        q = self.q_linear(q) # [batch_size, max_seq_len, d_model]
        k = self.q_linear(k) 
        v = self.q_linear(v)
        
        #head_numに分割
        q = self._split_head(q) # [batch_size, head_num, max_seq_len, d_model/head_num]
        k = self._split_head(k)
        v = self._split_head(v)
        
        #queryとkeyの関連度の計算と、Scaled Dot-production
        weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.d_model) # [batch_size, head_num, max_seq_len, max_seq_len]
        
        #maskをかける
        #multi-headを使う場合のmask
        #mask = mask.unsqueeze(1).unsqueeze(1)
        mask = mask.unsqueeze(1)
        weights = weights.masked_fill(mask==0, -1e9)# [batch_size, head_num, max_seq_len, max_seq_len]
        
        #AttentionWeightを計算
        attention_weight = F.softmax(weights, dim=-1)# [batch_size, head_num, q_length, k_length]
        
        #AttentionWeightよりvalueから情報を引き出す
        attention_output = torch.matmul(attention_weight, v)# [batch_size, head_num, q_length, d_model/head_num]
        attention_output = self._combine_head(attention_output)
        output = self.out(attention_output)
        
        
        return output, attention_weight
        
    def _split_head(self, x):
        """
        x.size:[batch_size, length, d_model]
        """
        batch_size, length, d_model = x.size()
        x = x.view(batch_size, length, self.head_num, self.d_model//self.head_num) #reshape
        return x.permute(0, 2, 1, 3)
    
    #outputする前に分割したheadを戻す。
    def _combine_head(self, x):
        """
        x.size:[batch_size, head_num, length, d_model//head_num]
        """
        batch_size, _, length, _  = x.size()
        x = x.permute(0, 2, 1, 3)
        return x.reshape(batch_size, length, self.d_model)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=1024, dropout=0.1):
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """
        x size=[batch_size, length, d_model]
        return size=[batch_size, length, d_model]
        """
        x = self.linear_1(x)
        x = self.dropout(F.relu(x))
        x = self.linear_2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, head_num, dropout=0.1):
        super().__init__()

        # LayerNormalization
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        # Attention
        self.attn = MultiheadAttention(d_model, head_num, dropout)
        # FFN
        self.ff = FeedForward(d_model)
        # Dropout
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        # SelfAttention
        x_normlized = self.norm_1(x)
        output, normlized_weights = self.attn(
            x_normlized, x_normlized, x_normlized, mask)
        x2 = x + self.dropout_1(output)
        # FFN
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


class TransformerEncoderClassification(nn.Module):

    def __init__(self, text_embedding_vectors, head_num, dropout=0.1, d_model=300, max_seq_len=140, output_dim=5):
        super().__init__()

        # モデル構築
        self.net1 = Embedder(text_embedding_vectors)
        self.net2 = PositionalEncoder(d_model=d_model, max_seq_len=max_seq_len)
        self.net3 = nn.Dropout(dropout)
        self.net4_1 = TransformerBlock(d_model=d_model, head_num=head_num, dropout=dropout)
        self.net4_2 = TransformerBlock(d_model=d_model, head_num=head_num, dropout=dropout)
        self.net5 = ClassificationHead(output_dim=output_dim, d_model=d_model)

    def forward(self, x, mask):
        x1 = self.net1(x)  #Embedding
        x2 = self.net2(x1) #PositinalEncoding
        x3 = self.net3(x2) #Dropout
        x4_1, normlized_weights_1 = self.net4_1(x3, mask) #self-Attention+FFN 
        x4_2, normlized_weights_2 = self.net4_2(x4_1, mask)  #self-Attention+FFN
        x5 = self.net5(x4_2)  #linear
        return x5, normlized_weights_1, normlized_weights_2


class Sonar(object):

    def __init__(self):
      emb = np.load('omomi0430.npy')
      emb = torch.from_numpy(emb.astype(np.float32)).clone()
      # index→sentence ex:itos[0] => '<cls>'
      self.itos = pickle.load(open('itos0430.pkl', 'rb')) 
      # sentence→index ex:stoi['<cls>'] => 0
      self.stoi = pickle.load(open('stoi0430.pkl', 'rb')) 
      model_path = 'net0430.pth'
      self.model = TransformerEncoderClassification(
        text_embedding_vectors=emb, head_num=5, d_model=300, max_seq_len=140, output_dim=4)
      self.model.load_state_dict(torch.load(model_path)) #モデルにパラメータを当てはめる
      self.model.eval()
      self.text = ''

    def make_html(self, text, index):
      "HTMLデータを作成する"
      #テキストを形態素解析
      if text != '':
        self.text = text
      mecab = MeCab.Tagger('-Owakati')
      print(self.text)
      result = [tok for tok in mecab.parse(self.text).split()]
      print(result)
      inputs = text_to_ids(result, self.stoi) # [140]
      inputs = inputs.unsqueeze(0) # [1, 140]
      #inputs = torch.tensor(inputs)
      
      # mask作成
      input_pad = 1  # <pad>は1
      input_mask = (inputs != input_pad)
      
      
      # 感情分類
      outputs, normlized_weights_1, normlized_weights_2 = self.model(
        inputs, input_mask)
      _, preds = torch.max(outputs, 1)  # ラベルを予測
      print(normlized_weights_1[0, :, 0, :])

      #label = batch.Label[index]  # ラベル
      #pred = preds  # 予測

      html = '<form action="/post" method="post" class="form-inline">'
      
      
      #index番目のデータの0番目のmemoryの関連度を抽出している
      
      #可視化するワード
      attn_word = self.itos[inputs[0, index]]
      attn_word = re.sub('<', '&lt;', attn_word)
      attn_word = re.sub('>', '&gt;', attn_word)
      html += '可視化ワード：{}<br><br>'.format(attn_word)
      #print(self.itos[inputs[index]])
      #0バッチ目のindex番目の単語
      
      #normlized_weights_1.shape -> [1, 5, 140, 140]
      attens1 = normlized_weights_1[0, :, index, :]
      #attens1.shape -> [5, 140]
      print(attens1)
      for i in range(5):
        attens1[i, :] /= attens1[i, :].max()
      
      """
      attens2 = normlized_weights_2[0, :, index, :]  # <cls>のAttention
      #attens2 /= attens2.max()
      print('attens2.shape', attens2.shape)
      for i in range(5):
        attens2[i, :] /= attens2[i, :].max()
      print('attens2.shape', attens2.shape)
      """
      # ラベルと予測結果を文字に置き換え
      #label_str = label
      #pred_str = pred

      
      # 表示用のHTMLを作成する
      html += '感情分析結果：{}<br><br>'.format(preds)
      
      #プルダウンメニューを作る
      html += '<select name="sel" class="form-control"><option value="null" disabled selected>分析ワードを選択</option>'
      for i, word in enumerate(result, 1):
        html+='<option value="{}">{}</option>'.format(i, word)
      html += '</select><br>'
      html += '<input type="text" class="form-control" id="name" name="name" placeholder="Name"><button type="submit" class="btn btn-default">送信する</button><br><br>'

      
      # 1段目のAttention
      html += '[{}のAttentionWeightを可視化(1段目)]<br>'.format(attn_word)
      #sentenceはid列
      
      """
      for word, attn in zip(inputs, attens1):
        html += highlight(self.itos[word], attn)
      html += "<br><br>"
      """
      html += '<table>'
      #attens1.shape -> [5, 140]
      #<cls>とwordのattention_weight
      for word, head in zip(inputs[0], attens1.transpose(1, 0)):
        #wordの各Headについて
        if word != 1:
          html += highlight(self.itos[word], head)
      html += "</table>"

      """
      # 2段目のAttention
      html += '[{}のAttentionWeightを可視化(2段目)]<br>'.format(attn_word)
      for word, attn in zip(inputs, attens2):
        html += highlight(self.itos[word], attn)
      """
      html += '</form>'

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

#色を設定
def highlight(word, head):
  #head=>[head_num]
  result = '<tr><td>{}</td>'.format(word)
  #各headについて
  for weight in head:
    html_color = '#%02X%02X%02X' % (255, int(255*(1 - weight)), int(255*(1 - weight)))
    result += '<td><span style="background-color: {}">#</span></td>'.format(html_color)
  return result + '</tr>'



