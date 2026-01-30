# Python 3.12 (安定版) を使用。3.14リリース後もベースイメージを変更するだけで対応可能。
FROM python:3.12-slim

# 作業ディレクトリの設定
WORKDIR /app

# 依存関係のインストールに必要な最小限のファイルをコピー
COPY requirements.txt .

# 依存関係のインストール
# --no-cache-dir でイメージサイズを削減
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコード全体をコピー
COPY . .

# 環境変数の読み込みを補助（コンテナ実行時に上書き可能）
ENV PORT=8000
ENV HOST=0.0.0.0

# ポートの公開
EXPOSE 8000

# サーバーの起動
# Render等のプラットフォームでは PORT 環境変数が渡されるため、$PORT を参照するようにする
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
