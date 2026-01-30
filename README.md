# 心理学RAGアシスタント

心理学の理論文書に基づいたAIチャットボットです。

## 📍 はじめに

このフォルダには必要なデータと設定が含まれています。
お使いのPCで以下の準備をすれば、すぐにチャットボットを利用できます。

## 🚀 準備と起動

### 1. Pythonのインストール
もしPCにPythonが入っていない場合は、インストールしてください。
- [Python公式サイト](https://www.python.org/downloads/)
- **重要**: インストール画面で **「Add Python to PATH」** に必ずチェックを入れてください。

### 2. ライブラリのインストール
このフォルダで「PowerShell」や「コマンドプロンプト」を開き、以下のコマンドを実行してください。
必要な機能が自動でインストールされます。

```powershell
pip install -r requirements.txt
```

### 3. APIキーの設定（重要）
1. フォルダ内にある `.env.example` ファイルをコピーして、名前を **`.env`** に変更してください。
2. 変更した `.env` ファイルをテキストエディタで開き、`OPENAI_API_KEY=` の後にあなたのAPIキーを貼り付けて保存してください。

### 4. アプリの起動
以下のコマンドを実行すると、チャットボットが起動します。
※データベース（`data/chromadb`）や文書データは同梱されているため、そのまま使用可能です。

```powershell
python -m uvicorn app.main:app --reload
```

黒い画面に `Application startup complete` と表示されたら起動完了です。
ブラウザで以下のURLを開いてください。

👉 **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)**

---

### 終了方法
終了するときは、ターミナルで `Ctrl + C` を押してください。

---

## 🌐 サーバーへのデプロイ（オープンURLの取得）

GitHubとRender.comを連携させることで、自分専用のURLで公開できます。

1. **GitHubへアップロード**: このフォルダをGitHubのリポジトリにプッシュしてください。
2. **Render.comでデプロイ**:
   - [Render](https://render.com/)にログインし、**New > Web Service** を選択。
   - GitHubリポジトリを選択。
   - **Environment Variables** に `OPENAI_API_KEY` を設定してください。
   - `render.yaml` が同梱されているため、設定は自動的に読み込まれます。
3. **完了**: 数分待つと「https://your-app.onrender.com」のようなURLが発行されます。

## 🐍 Python 3.14 への対応について

このプロジェクトは Python 3.11〜3.13 で動作確認されていますが、将来の **Python 3.14** でも動作するように設計されています。

- **ローカルでの実行**: `requirements.txt` のバージョン指定を緩和しているため、最新の Python 環境でもインストールが可能です。
- **Dockerでの実行**: 同梱の `Dockerfile` を使用することで、OSやPythonバージョンを問わず同じ環境で動作させることができます（推奨）。

---

## ⚠️ トラブルシューティング
（中略）
