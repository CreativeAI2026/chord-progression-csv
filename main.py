# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 12:39:45 2026

@author: nacht
"""

import os
import re
import random
import numpy as np
 
# --------------------
# 移調（トランスポーズ）のための準備
# --------------------
 
NOTE_TO_PC = {
    "C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11
}
 
PC_TO_NOTE = [
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B"
]
 
def note_to_pc(root_str):
    letter = root_str[0]
    accidental = root_str[1:] if len(root_str) > 1 else ""
 
    pc = NOTE_TO_PC[letter]
 
    if accidental == "#":
        pc += 1
    elif accidental in ("b", "-"):
        pc -= 1
 
    return pc % 12
 
 
def transpose_chord(chord_str, semitone_shift):
    root_str, quality = chord_str.split(":")
    pc = note_to_pc(root_str)
    new_pc = (pc + semitone_shift) % 12
    new_root = PC_TO_NOTE[new_pc]
    return f"{new_root}:{quality}"
 
 
# --------------------
# データ読み込み（トニックを問わず全曲を使用し、Cに移調する）
# --------------------
 
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "McGill-Billboard")
 
all_songs = []
 
for song_folder in os.listdir(root):
 
    folder_path = os.path.join(root, song_folder)
 
    if not os.path.isdir(folder_path):
        continue
 
    txt_path = os.path.join(folder_path, "salami_chords.txt")
 
    if not os.path.exists(txt_path):
        continue
 
    with open(txt_path, encoding="utf-8") as f:
        text = f.read()
 
    tonic_match = re.search(
        r'# tonic:\s*([A-G][#b]?)',
        text
    )
 
    if not tonic_match:
        continue
 
    tonic = tonic_match.group(1)
 
    # コードクオリティを以下11種類に拡張して抽出する
    # メジャー(maj) / マイナー(min) / ディミニッシュ(dim) / オーギュメント(aug)
    # メジャーセブンス(maj7) / マイナーセブンス(min7) / ドミナントセブンス(7)
    # ハーフディミニッシュ(hdim7) / ディミニッシュセブンス(dim7)
    # サス2(sus2) / サス4(sus4)
    # ※ "maj7"が"maj"に途中で切られないよう、長い表記を先に判定する順序にしている
    chords = re.findall(
        r'([A-G][#b-]?:(?:maj7|min7|hdim7|dim7|maj6|min6|maj9|min9|sus4|sus2|maj|min|dim|aug|7|6|9))',
        text
    )
 
    if len(chords) == 0:
        continue
 
    tonic_pc = note_to_pc(tonic)
    shift = (0 - tonic_pc) % 12
 
    transposed_chords = [
        transpose_chord(chord, shift)
        for chord in chords
    ]
 
    all_songs.append(transposed_chords)
 
print("曲数:", len(all_songs))
 
# --------------------
# 重複削除
# --------------------
 
compressed_songs = []
 
for song in all_songs:
 
    compressed = []
 
    for chord in song:
 
        if len(compressed) == 0 or chord != compressed[-1]:
            compressed.append(chord)
 
    compressed_songs.append(compressed)
 
# --------------------
# コード辞書作成
# --------------------
 
from collections import Counter
 
all_chords = []
 
for song in compressed_songs:
    all_chords.extend(song)
 
counter = Counter(all_chords)
 
unique_chords = sorted(counter.keys())
 
print("コード種類数:", len(unique_chords))
 
# --------------------
# クオリティ別の出現数を確認（診断用）
# --------------------
 
quality_jp_names = {
    "maj": "メジャー",
    "min": "マイナー",
    "dim": "ディミニッシュ",
    "aug": "オーギュメント",
    "maj7": "メジャーセブンス",
    "min7": "マイナーセブンス",
    "7": "ドミナントセブンス",
    "hdim7": "ハーフディミニッシュ",
    "dim7": "ディミニッシュセブンス",
    "sus2": "サス2",
    "sus4": "サス4",
    "maj9": "メジャーナインス",
    "min9": "マイナーナインス",
    "9": "ナインス",
    "maj6": "メジャーシックス",
    "min6": "マイナーシックス",
    "6": "シックス",
}
 
quality_counter = Counter()
 
for chord in all_chords:
    quality = chord.split(":")[1]
    quality_counter[quality] += 1
 
print()
print("--- クオリティ別の出現数 ---")
for quality, jp_name in quality_jp_names.items():
    count = quality_counter.get(quality, 0)
    print(f"{jp_name:<14}({quality:<6}): {count}回")
print()
 
chord_to_idx = {
    chord:i
    for i, chord in enumerate(unique_chords)
}
 
idx_to_chord = {
    i:chord
    for chord, i in chord_to_idx.items()
}
 
# --------------------
# 数値化
# --------------------
 
songs_encoded = []
 
for song in compressed_songs:
 
    encoded = [
        chord_to_idx[chord]
        for chord in song
    ]
 
    songs_encoded.append(encoded)
 
# --------------------
# 学習データ作成
# --------------------
 
SEQ_LEN = 8
 
X = []
y = []
 
for song in songs_encoded:
 
    if len(song) <= SEQ_LEN:
        continue
 
    for i in range(len(song) - SEQ_LEN):
 
        X.append(song[i:i+SEQ_LEN])
        y.append(song[i+SEQ_LEN])
 
X = np.array(X)
y = np.array(y)
 
print("学習データ数:", len(X))
 
# --------------------
# LSTM
# --------------------
 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
 
num_chords = len(unique_chords)
 
model = Sequential([
    Embedding(
        input_dim=num_chords,
        output_dim=32
    ),
 
    LSTM(128),
 
    Dense(
        num_chords,
        activation="softmax"
    )
])
 
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)
 
# --------------------
# 学習
# --------------------
 
history = model.fit(
    X,
    y,
    epochs=30,
    batch_size=32,
    validation_split=0.1
)
 
# --------------------
# 確率的サンプリング用の関数（バッチ対応版）
# --------------------
 
def sample_with_temperature_batch(pred_batch, temperature=1.0):
    """
    pred_batch: shape = (バッチ数, コード種類数) のsoftmax確率
    バッチ内の各行ごとに、確率分布に従って1つインデックスを選ぶ。
    戻り値: shape = (バッチ数,) の選ばれたインデックス配列
    """
 
    pred_batch = np.asarray(pred_batch).astype("float64")
    pred_batch = np.log(pred_batch + 1e-8) / temperature
 
    exp_pred = np.exp(pred_batch)
    probs = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
 
    chosen = np.array([
        np.random.choice(probs.shape[1], p=probs[i])
        for i in range(probs.shape[0])
    ])
 
    return chosen
 
 
# --------------------
# 8コード生成（大量出力・バッチ処理版）
# --------------------
 
NUM_GENERATIONS = 500   # ここを増やすほど多くのパターンを生成する
TEMPERATURE = 1.5
GEN_LEN = 8
 
# NUM_GENERATIONS個ぶんのseedを一度にまとめて選ぶ
seed_indices = np.random.choice(len(X), size=NUM_GENERATIONS, replace=True)
generated_batch = X[seed_indices].tolist()  # shape: (NUM_GENERATIONS, SEQ_LEN)
 
for step in range(GEN_LEN):
 
    # 直近SEQ_LEN個を取り出し、バッチとしてまとめてモデルに入力
    x_batch = np.array([
        seq[-SEQ_LEN:] for seq in generated_batch
    ])
 
    # 1回のpredict呼び出しでNUM_GENERATIONS件すべてを予測(高速)
    pred_batch = model.predict(
        x_batch,
        verbose=0
    )
 
    next_chords = sample_with_temperature_batch(
        pred_batch,
        temperature=TEMPERATURE
    )
 
    for i in range(NUM_GENERATIONS):
        generated_batch[i].append(int(next_chords[i]))
 
# 各系列の最後のGEN_LEN個を取り出してコード名に変換
results = []
 
for seq in generated_batch:
 
    result = [
        idx_to_chord[i]
        for i in seq[-GEN_LEN:]
    ]
 
    results.append(result)
 
print(f"生成完了: {len(results)}パターン")
 
# --------------------
# 重複状況の確認
# --------------------
 
results_tuples = [tuple(r) for r in results]
unique_count = len(set(results_tuples))
duplicate_count = len(results_tuples) - unique_count
 
print(f"ユニークなパターン数: {unique_count} / {len(results_tuples)}")
print(f"重複しているパターン数: {duplicate_count}")
print(f"重複率: {duplicate_count / len(results_tuples) * 100:.1f}%")
 
# 先頭5件だけ画面にも表示（全部表示すると見づらいため）
for n, result in enumerate(results[:5]):
    print()
    print("パターン", n + 1)
    print(result)
 
# --------------------
# CSV書き出し（全パターンを保存）
# --------------------
 
import csv
 
with open(
    "generated_chords.csv",
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:
 
    writer = csv.writer(f)
 
    writer.writerow([
        "Chord1",
        "Chord2",
        "Chord3",
        "Chord4",
        "Chord5",
        "Chord6",
        "Chord7",
        "Chord8"
    ])
 
    writer.writerows(results)
 
print("generated_chords.csv に全パターンを書き出しました。")
 