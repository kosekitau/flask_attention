# Flask などの必要なライブラリをインポートする
from flask import Flask, render_template, request, redirect, url_for
from api import Sonar

# 自身の名称を app という名前でインスタンス化する
sonar = Sonar()
app = Flask(__name__)

# 最初の画面
@app.route('/')
def index():
    title = "感情分析"
    message = "テキストを入力"
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
  #分析ワードを選択した場合
  if request.form.get('sel') != None:
    #から分析ワードのindexを取得
    index = int(request.form.get('sel'))
    sonar.make_html('', index)
    return render_template('test.html')
  
  elif request.form.get('text') != None:
    # テキストボックスから分類する文章を取得
    text = request.form.get('text')
    #text = request.form.get('radio')
    #感情分析を行い、test.html をレンダリングする
    sonar.make_html(text, 0)
    return render_template('test.html')
      
  
  else:
    print(request.form.get('sel'))
    print(request.form.get('name'))
    # エラーなどでリダイレクトしたい場合はこんな感じで
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.debug = True # デバッグモード有効化
    app.run(host='0.0.0.0') # どこからでもアクセス可能に