# Flask などの必要なライブラリをインポートする
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from api import Sonar

# 自身の名称を app という名前でインスタンス化する
sonar = Sonar()
app = Flask(__name__)

# メッセージをランダムに表示するメソッド
def picked_up():
    messages = [
        "こんにちは、あなたの名前を入力してください",
        "やあ！お名前は何ですか？",
        "あなたの名前を教えてね"
    ]
    # NumPy の random.choice で配列からランダムに取り出し
    return np.random.choice(messages)

# 最初の画面
@app.route('/')
def index():
    title = "感情分析"
    message = picked_up()
    # messageをindex.htmlのmessageへ
    return render_template('index.html',
                           message=message, title=title)

# /post にアクセスしたときの処理
@app.route('/post', methods=['GET', 'POST'])
def post():
  title = "こんにちは"
  #ボタンを押すとrequestが来るっぽい？
  #resquest.form['name']に入力したテキストが入ってる？
  #getで拾えない場合はエラーを吐かずNoneを渡す
  #新しく入力されたテキストを分析
  if request.form.get('sel') == None:
    # テキストボックスから分類する文章を取得
    text = request.form.get('name')
    #text = request.form.get('radio')
    #感情分析を行い、test.html をレンダリングする
    sonar.make_html(text, 0)
    return render_template('test.html')
      
  #分析ワードを選択した場合
  elif request.form.get('name') == None:
    #から分析ワードのindexを取得
    index = int(request.form.get('sel'))
    sonar.make_html('', index)
    return render_template('test.html')
  
  else:
    # エラーなどでリダイレクトしたい場合はこんな感じで
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.debug = True # デバッグモード有効化
    app.run(host='0.0.0.0') # どこからでもアクセス可能に