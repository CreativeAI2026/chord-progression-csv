# chord-progression-csv

McGill Billboard データセットのコード進行を LSTM で学習し、生成した 8 コードの進行を `generated_chords.csv` に書き出します。

## セットアップ

### リポジトリを取得

```bash
git clone https://github.com/CreativeAI2026/chord-progression-csv.git
cd chord-progression-csv
```

以降のコマンドはすべてこのフォルダで実行します。

### Python 3.13 をインストール

TensorFlow が対応しているのは Python 3.10〜3.13 なので、3.13 を使います（最新の 3.14 では動きません）。

macOS（[Homebrew](https://brew.sh/ja/) が必要）:

```bash
brew install python@3.13
```

Windows（PowerShell）:

```powershell
winget install Python.Python.3.13
```

インストール後、ターミナルを開き直してください。

### ライブラリをインストール

macOS:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows（PowerShell）:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### データセットを用意

AI に学習させるコード進行のデータ（[McGill Billboard](https://ddmal.ca/research/The_McGill_Billboard_Project_(Chord_Analysis_Dataset)/)、ビルボードのヒット曲 約 900 曲分）をダウンロードします。

macOS:

```bash
curl -L -o billboard.tar.gz "https://www.dropbox.com/s/2lvny9ves8kns4o/billboard-2.0-salami_chords.tar.gz?dl=1"
tar -xzf billboard.tar.gz
```

Windows（PowerShell）:

```powershell
curl.exe -L -o billboard.tar.gz "https://www.dropbox.com/s/2lvny9ves8kns4o/billboard-2.0-salami_chords.tar.gz?dl=1"
tar -xzf billboard.tar.gz
```

`main.py` と同じ場所に `McGill-Billboard` フォルダができていれば OK です。

## 実行

```bash
python main.py
```

生成数などは `main.py` 内の `NUM_GENERATIONS`（生成数）や `TEMPERATURE`（多様さ）で調整できます。
