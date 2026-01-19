# proconio とは

`proconio`は、AtCoderなどの競技プログラミングでRustを使う際に**必須**の入力ライブラリです。

## 📚 概要

- **正式名称**: `proconio` (Programming Contest Input/Output)
- **目的**: 競技プログラミング用の高速で使いやすい入力マクロ
- **開発**: AtCoderのRustコミュニティで開発・メンテナンス
- **特徴**: 型推論が強力で、複雑な入力も簡潔に書ける

## 🚀 なぜ必要？

### 標準入力の問題点

```rust
// ❌ 標準ライブラリだけだと面倒
use std::io;

fn main() {
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    let n: usize = input.trim().parse().unwrap();
    
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    let values: Vec<i32> = input
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
}
```

### proconioを使うと

```rust
// ✅ proconioなら1行で書ける
use proconio::input;

fn main() {
    input! {
        n: usize,
        values: [i32; n],
    }
}
```

**圧倒的に簡潔！**

---

## 📦 インストール

`Cargo.toml`に追加：

```toml
[dependencies]
proconio = { version = "0.4", features = ["derive"] }
```

**`features = ["derive"]`は必須** - これがないと`input!`マクロが使えません。

---

## 🎯 基本的な使い方

### 1. 単純な入力

```rust
use proconio::input;

fn main() {
    input! {
        n: usize,      // 1つの整数
        s: String,     // 1つの文字列
    }
    
    println!("n = {}, s = {}", n, s);
}
```

**入力例**:
```
5
hello
```

**出力**:
```
n = 5, s = hello
```

### 2. 複数の値を一度に

```rust
use proconio::input;

fn main() {
    input! {
        a: i32,
        b: i32,
        c: i32,
    }
    
    println!("{}", a + b + c);
}
```

**入力例**:
```
1 2 3
```

**出力**:
```
6
```

### 3. 配列の入力

```rust
use proconio::input;

fn main() {
    input! {
        n: usize,
        a: [i32; n],  // サイズnの配列
    }
    
    for &x in &a {
        println!("{}", x);
    }
}
```

**入力例**:
```
3
10 20 30
```

**出力**:
```
10
20
30
```

### 4. 固定サイズの配列

```rust
use proconio::input;

fn main() {
    input! {
        a: [i32; 3],  // サイズ3の固定配列
    }
    
    println!("{:?}", a);
}
```

**入力例**:
```
1 2 3
```

**出力**:
```
[1, 2, 3]
```

### 5. 2次元配列（行列）

```rust
use proconio::input;

fn main() {
    input! {
        h: usize,
        w: usize,
        grid: [[char; w]; h],  // h×wの2次元配列
    }
    
    for row in &grid {
        for &cell in row {
            print!("{}", cell);
        }
        println!();
    }
}
```

**入力例**:
```
2 3
abc
def
```

**出力**:
```
abc
def
```

### 6. タプル

```rust
use proconio::input;

fn main() {
    input! {
        n: usize,
        points: [(i32, i32); n],  // (x, y)のタプルの配列
    }
    
    for (x, y) in &points {
        println!("({}, {})", x, y);
    }
}
```

**入力例**:
```
2
1 2
3 4
```

**出力**:
```
(1, 2)
(3, 4)
```

---

## 🔥 高度な使い方

### 1. 構造体で入力を受け取る

```rust
use proconio::input;

#[derive(Debug)]
struct Point {
    x: i32,
    y: i32,
}

fn main() {
    input! {
        n: usize,
        points: [Point; n],
    }
    
    // これはエラー！Pointの構造体を直接受け取れない
    // 代わりにタプルを使う
}
```

**正しい方法**:

```rust
use proconio::input;

fn main() {
    input! {
        n: usize,
        points: [(i32, i32); n],  // タプルで受け取る
    }
    
    // 必要に応じて構造体に変換
    let points: Vec<Point> = points
        .into_iter()
        .map(|(x, y)| Point { x, y })
        .collect();
}
```

### 2. 文字列の配列

```rust
use proconio::input;

fn main() {
    input! {
        n: usize,
        s: [String; n],  // 文字列の配列
    }
    
    for str in &s {
        println!("{}", str);
    }
}
```

**入力例**:
```
3
hello
world
rust
```

### 3. 複数行の入力パターン

```rust
use proconio::input;

fn main() {
    input! {
        n: usize,
        m: usize,
        edges: [(usize, usize); m],  // グラフの辺
    }
    
    // グラフを構築
    let mut graph = vec![vec![]; n];
    for (u, v) in edges {
        graph[u].push(v);
        graph[v].push(u);
    }
}
```

---

## 📊 対応している型

| 型 | 説明 | 例 |
|---|---|---|
| `i8`, `i16`, `i32`, `i64`, `i128` | 符号付き整数 | `input! { n: i32 }` |
| `u8`, `u16`, `u32`, `u64`, `u128` | 符号なし整数 | `input! { n: usize }` |
| `f32`, `f64` | 浮動小数点数 | `input! { x: f64 }` |
| `char` | 1文字 | `input! { c: char }` |
| `String` | 文字列 | `input! { s: String }` |
| `[T; n]` | 固定サイズ配列 | `input! { a: [i32; 3] }` |
| `[T]` | 可変長配列 | `input! { n: usize, a: [i32; n] }` |
| `[[T; w]; h]` | 2次元配列 | `input! { grid: [[char; w]; h] }` |
| `(T1, T2, ...)` | タプル | `input! { p: (i32, i32) }` |

---

## ⚡ パフォーマンス

`proconio`は**非常に高速**です：

- **内部実装**: `BufReader`と`Vec`を使った最適化されたパーサー
- **メモリ効率**: 必要最小限のメモリ使用
- **AtCoder環境**: AtCoderのRust環境で最適化されている

**ベンチマーク例**（100万個の整数を読み込む）:
- 標準ライブラリ: ~200ms
- `proconio`: ~50ms

---

## 🎯 実践的な例

### AtCoder典型90問: 001 - Yokan Party

```rust
use proconio::input;

fn main() {
    input! {
        n: usize,
        l: i32,
        k: usize,
        a: [i32; n],
    }
    
    // 二分探索で解く
    let mut left = 0;
    let mut right = l;
    
    while right - left > 1 {
        let mid = (left + right) / 2;
        if can_cut(&a, l, k, mid) {
            left = mid;
        } else {
            right = mid;
        }
    }
    
    println!("{}", left);
}

fn can_cut(a: &[i32], l: i32, k: usize, min_len: i32) -> bool {
    let mut last = 0;
    let mut count = 0;
    
    for &x in a {
        if x - last >= min_len && l - x >= min_len {
            count += 1;
            last = x;
        }
    }
    
    count >= k
}
```

### グラフ問題の入力

```rust
use proconio::input;

fn main() {
    input! {
        n: usize,  // 頂点数
        m: usize,  // 辺数
        edges: [(usize, usize); m],  // 辺のリスト
    }
    
    // 無向グラフを構築
    let mut graph = vec![vec![]; n];
    for (u, v) in edges {
        graph[u].push(v);
        graph[v].push(u);
    }
    
    // BFSなどで処理
}
```

---

## ⚠️ よくある間違い

### 1. `features = ["derive"]`を忘れる

```toml
# ❌ エラーになる
proconio = "0.4"

# ✅ 正しい
proconio = { version = "0.4", features = ["derive"] }
```

### 2. 配列のサイズを間違える

```rust
// ❌ エラー: サイズが一致しない
input! {
    n: usize,
    a: [i32; 5],  // n=3なのに5個読み込もうとする
}

// ✅ 正しい
input! {
    n: usize,
    a: [i32; n],  // n個読み込む
}
```

### 3. 型を間違える

```rust
// ❌ オーバーフローの可能性
input! {
    n: i32,  // 10^9を超える可能性がある
}

// ✅ 正しい
input! {
    n: i64,  // またはusize
}
```

---

## 🔗 関連リソース

- [proconio公式ドキュメント](https://docs.rs/proconio/)
- [AtCoder Rust環境](https://atcoder.jp/contests/abs/submissions/me)
- [proconioのGitHub](https://github.com/stateless7/proconio-rs)

---

## 📝 まとめ

`proconio`は：

1. **簡潔**: 1行で複雑な入力も書ける
2. **高速**: 最適化されたパーサー
3. **型安全**: コンパイル時に型チェック
4. **AtCoder標準**: ほとんどのRustユーザーが使用

AtCoderでRustを使うなら、`proconio`は**必須**です！

