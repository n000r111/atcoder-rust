use proconio::input;
use rand::prelude::*;
use rand::thread_rng;
use std::collections::BTreeSet;
use std::time::Instant;

// スコア計算関数
fn evaluate_score(
    _n: usize,
    k: usize,
    adj: &[Vec<usize>],
    ops: &[i32],
    initial_ice_type: &[char],
) -> i64 {
    let mut ice_type = initial_ice_type.to_vec();
    let mut pos = 0;
    let mut prev: Option<usize> = None;
    let mut cone = Vec::new();
    let mut shops: Vec<BTreeSet<Vec<char>>> = vec![BTreeSet::new(); k];

    for &op in ops {
        match op {
            -1 => {
                // Flip操作
                if pos >= k && ice_type[pos] == 'W' {
                    ice_type[pos] = 'R';
                }
            }
            next_idx => {
                let next = next_idx as usize;
                // 前回の位置に戻れないチェック
                if let Some(prev_pos) = prev {
                    if next == prev_pos {
                        return 0; // 無効な操作
                    }
                }
                // 隣接チェック
                if !adj[pos].contains(&next) {
                    return 0; // 無効な操作
                }
                prev = Some(pos);
                pos = next;

                if pos < k {
                    // ショップ: 納品
                    shops[pos].insert(cone.clone());
                    cone.clear();
                } else {
                    // 木: 収穫
                    cone.push(ice_type[pos]);
                }
            }
        }
    }

    // スコア計算
    shops.iter().map(|s| s.len() as i64).sum()
}

fn main() {
    input! {
        n: usize,
        m: usize,
        k: usize,
        t: usize,
        edges: [(usize, usize); m],
        _vertices: [(i32, i32); n],
    }

    // 隣接リストを構築
    let mut adj = vec![vec![]; n];
    for &(u, v) in &edges {
        adj[u].push(v);
        adj[v].push(u);
    }

    // 状態管理
    let mut ice_type = vec!['W'; n]; // 各木のアイスタイプ（初期はすべてW）
    for i in 0..k {
        ice_type[i] = 'S'; // ショップは'S'でマーク（使用しない）
    }

    // 各ショップから到達可能な木を探索（BFS）
    let mut reachable_trees: Vec<Vec<usize>> = vec![vec![]; k];
    for shop_id in 0..k {
        let mut visited = vec![false; n];
        let mut queue = vec![shop_id];
        visited[shop_id] = true;

        while let Some(v) = queue.pop() {
            if v >= k {
                reachable_trees[shop_id].push(v);
            }
            for &next in &adj[v] {
                if !visited[next] {
                    visited[next] = true;
                    queue.push(next);
                }
            }
        }
    }

    let start_time = Instant::now();
    let initial_solution_time_limit = 1.8; // 初期解生成に使う時間

    let mut rng = thread_rng();
    let mut best_initial: Option<(Vec<i32>, i64)> = None;

    // 複数の初期解を生成して最良のものを選ぶ
    while start_time.elapsed().as_secs_f64() < initial_solution_time_limit {
        // 状態管理
        let mut current = 0;
        let mut prev_pos: Option<usize> = None;
        let mut ice_type_local = ice_type.clone();
        let mut cone = Vec::new();
        let mut shops_local: Vec<BTreeSet<Vec<char>>> = vec![BTreeSet::new(); k];
        let mut output_local: Vec<i32> = vec![];

        // グラフ構造を活用した戦略で初期解を生成
        for _step in 0..t {
            // 移動可能な頂点を探す
            let mut next_candidates: Vec<usize> = adj[current]
                .iter()
                .filter(|&&next| prev_pos.map(|p| next != p).unwrap_or(true))
                .copied()
                .collect();

            if next_candidates.is_empty() {
                break;
            }

            // 新たな組み合わせが作れない場合、Flip操作を試みる
            if !cone.is_empty() {
                let all_shops_have_it = (0..k).all(|shop_id| shops_local[shop_id].contains(&cone));

                if all_shops_have_it && current >= k && ice_type_local[current] == 'W' {
                    let can_move = adj[current]
                        .iter()
                        .any(|&next| prev_pos.map(|p| next != p).unwrap_or(true));

                    if can_move {
                        output_local.push(-1);
                        ice_type_local[current] = 'R';
                        continue;
                    }
                }
            }

            // コーンの内容が既にショップに納品されている場合、そのショップを避ける
            let shop_candidates: Vec<usize> = next_candidates
                .iter()
                .filter(|&&v| v < k)
                .filter(|&&shop_id| !shops_local[shop_id].contains(&cone))
                .copied()
                .collect();

            if !shop_candidates.is_empty() {
                next_candidates = shop_candidates;
            } else if !cone.is_empty() {
                next_candidates.retain(|&v| v >= k);
                if next_candidates.is_empty() {
                    next_candidates = adj[current]
                        .iter()
                        .filter(|&&next| prev_pos.map(|p| next != p).unwrap_or(true))
                        .copied()
                        .collect();
                }
            } else {
                let tree_candidates: Vec<usize> = next_candidates
                    .iter()
                    .filter(|&&v| v >= k)
                    .copied()
                    .collect();

                if !tree_candidates.is_empty() {
                    // より多くのショップから到達可能な木を優先
                    // 各木について、その木に到達可能なショップのうち、まだ未納品の組み合わせが多いショップの数をカウント
                    let mut scored_trees: Vec<(usize, usize)> = tree_candidates
                        .iter()
                        .map(|&tree| {
                            let score = (0..k)
                                .filter(|&shop_id| {
                                    reachable_trees[shop_id].contains(&tree)
                                        && shops_local[shop_id].len()
                                            < reachable_trees[shop_id].len()
                                })
                                .count();
                            (tree, score)
                        })
                        .collect();

                    // スコアが高い順にソート
                    scored_trees.sort_by(|a, b| b.1.cmp(&a.1));

                    // スコアが高い木を優先（同じスコアの場合はランダム）
                    if !scored_trees.is_empty() && scored_trees[0].1 > 0 {
                        let max_score = scored_trees[0].1;
                        let best_trees: Vec<usize> = scored_trees
                            .iter()
                            .take_while(|(_, score)| *score == max_score)
                            .map(|(tree, _)| *tree)
                            .collect();
                        next_candidates = best_trees;
                    } else {
                        next_candidates = tree_candidates;
                    }
                } else {
                    next_candidates = adj[current]
                        .iter()
                        .filter(|&&next| prev_pos.map(|p| next != p).unwrap_or(true))
                        .copied()
                        .collect();
                }
            }

            let next = next_candidates[rng.gen_range(0..next_candidates.len())];

            output_local.push(next as i32);
            prev_pos = Some(current);
            current = next;

            if current < k {
                shops_local[current].insert(cone.clone());
                cone.clear();
            } else {
                cone.push(ice_type_local[current]);
            }
        }

        // 初期解のスコアを評価
        let score = evaluate_score(n, k, &adj, &output_local, &ice_type);
        if best_initial.is_none() || score > best_initial.as_ref().unwrap().1 {
            best_initial = Some((output_local, score));
        }
    }

    let output = best_initial.unwrap_or_else(|| (vec![], 0)).0;

    // 出力
    for op in output {
        if op == -1 {
            print!("-1 ");
        } else {
            print!("{} ", op);
        }
    }
    println!();
}
