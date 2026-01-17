# AtCoder リポジトリ

AtCoderの問題を解くためのリポジトリです。

## ディレクトリ構造

```
atcoder/
├── alg/          # アルゴリズムコンテスト用
│   ├── compete.toml
│   └── {contest}/  # コンテストごとのディレクトリ
│       ├── Cargo.toml
│       └── src/
│           └── bin/
│               └── {problem}.rs
│
└── hrc/          # Heuristic Contest用
    ├── compete.toml
    └── {contest}/  # コンテストごとのディレクトリ
        ├── Cargo.toml
        └── src/
            └── bin/
                └── {problem}.rs
```

## 使い方

### アルゴリズムコンテスト

```bash
# alg/ ディレクトリに移動
cd alg

# コンテストを取得（例: ABC100）
cargo compete new abc100

# 問題をテスト
cargo compete test abc100-a

# 提出
cargo compete submit abc100-a
```

### Heuristic Contest

```bash
# hrc/ ディレクトリに移動
cd hrc

# コンテストを取得（例: AHC001）
cargo compete new ahc001

# 問題をテスト
cargo compete test ahc001-a

# 提出
cargo compete submit ahc001-a
```

## 設定の違い

- **alg/**: 基本的なアルゴリズム問題用。`proconio`のみ。
- **hrc/**: ヒューリスティック問題用。`proconio`に加えて`rand`と`rand_pcg`を含む。

## 注意事項

- 各ディレクトリ（`alg/`、`hrc/`）で`cargo compete`コマンドを実行してください。
- コンテストごとにディレクトリが作成されます。

