use proconio::input;
use rand::prelude::*;
use rand::thread_rng;
use std::collections::BTreeSet;
use std::time::Instant;

// ========================================
// 状態スナップショット用データ構造
// ========================================

/// 納品イベントの記録
#[derive(Clone, Debug)]
struct DeliveryDelta {
    shop_id: usize,
    combination: Vec<char>,
}

/// 1ステップの変更を表す差分
#[derive(Clone, Debug)]
enum StepDelta {
    Move {
        from: usize,
        to: usize,
        delivery: Option<DeliveryDelta>,
        harvested: Option<char>,
    },
    Flip {
        node: usize,
    },
}

/// フルスナップショット（チェックポイント）
#[derive(Clone)]
struct Checkpoint {
    step: usize,
    pos: usize,
    prev: Option<usize>,
    ice_type: Vec<char>,
    cone: Vec<char>,
    shops: Vec<BTreeSet<Vec<char>>>,
}

impl Checkpoint {
    fn new(n: usize, k: usize) -> Self {
        let mut ice_type = vec!['W'; n];
        for i in 0..k {
            ice_type[i] = 'S';
        }
        Self {
            step: 0,
            pos: 0,
            prev: None,
            ice_type,
            cone: Vec::new(),
            shops: vec![BTreeSet::new(); k],
        }
    }
}

/// 状態履歴管理
struct StateHistory {
    checkpoints: Vec<Checkpoint>,
    deltas: Vec<StepDelta>,
    checkpoint_interval: usize,
    #[allow(dead_code)]
    k: usize,
    #[allow(dead_code)]
    n: usize,
}

impl StateHistory {
    fn new(n: usize, k: usize, checkpoint_interval: usize) -> Self {
        let initial_checkpoint = Checkpoint::new(n, k);
        Self {
            checkpoints: vec![initial_checkpoint],
            deltas: Vec::new(),
            checkpoint_interval,
            k,
            n,
        }
    }

    /// 指定ステップまでの状態を復元
    fn restore_to_step(&self, target_step: usize) -> Checkpoint {
        // target_step以前の最も近いチェックポイントを見つける
        let checkpoint_idx = (target_step / self.checkpoint_interval).min(self.checkpoints.len() - 1);
        let mut state = self.checkpoints[checkpoint_idx].clone();
        let start_step = state.step;

        // チェックポイントからtarget_stepまでの差分を適用
        for i in start_step..target_step {
            if i >= self.deltas.len() {
                break;
            }
            match &self.deltas[i] {
                StepDelta::Move {
                    from,
                    to,
                    delivery,
                    harvested,
                } => {
                    state.prev = Some(*from);
                    state.pos = *to;

                    if let Some(d) = delivery {
                        state.shops[d.shop_id].insert(d.combination.clone());
                        state.cone.clear();
                    }
                    if let Some(ice) = harvested {
                        state.cone.push(*ice);
                    }
                }
                StepDelta::Flip { node } => {
                    state.ice_type[*node] = 'R';
                }
            }
        }
        state.step = target_step;
        state
    }

    /// 現在の差分数を取得
    fn len(&self) -> usize {
        self.deltas.len()
    }
}

/// 解からStateHistoryを構築
fn build_history_from_solution(
    n: usize,
    k: usize,
    adj: &[Vec<usize>],
    initial_ice_type: &[char],
    solution: &[i32],
    checkpoint_interval: usize,
) -> StateHistory {
    let mut history = StateHistory::new(n, k, checkpoint_interval);
    let mut pos = 0usize;
    let mut prev: Option<usize> = None;
    let mut ice_type = initial_ice_type.to_vec();
    let mut cone: Vec<char> = Vec::new();
    let mut shops: Vec<BTreeSet<Vec<char>>> = vec![BTreeSet::new(); k];

    for (step, &op) in solution.iter().enumerate() {
        match op {
            -1 => {
                if pos >= k && ice_type[pos] == 'W' {
                    history.deltas.push(StepDelta::Flip { node: pos });
                    ice_type[pos] = 'R';
                }
            }
            next_idx => {
                let next = next_idx as usize;
                // 隣接チェック（無効な場合はスキップ）
                if !adj[pos].contains(&next) {
                    continue;
                }
                if let Some(prev_pos) = prev {
                    if next == prev_pos {
                        continue;
                    }
                }

                let delivery = if next < k {
                    let d = DeliveryDelta {
                        shop_id: next,
                        combination: cone.clone(),
                    };
                    shops[next].insert(cone.clone());
                    cone.clear();
                    Some(d)
                } else {
                    None
                };

                let harvested = if next >= k {
                    let h = ice_type[next];
                    cone.push(h);
                    Some(h)
                } else {
                    None
                };

                history.deltas.push(StepDelta::Move {
                    from: pos,
                    to: next,
                    delivery,
                    harvested,
                });
                prev = Some(pos);
                pos = next;
            }
        }

        // チェックポイント間隔ごとにスナップショットを保存
        if (step + 1) % checkpoint_interval == 0 {
            let checkpoint = Checkpoint {
                step: step + 1,
                pos,
                prev,
                ice_type: ice_type.clone(),
                cone: cone.clone(),
                shops: shops.clone(),
            };
            history.checkpoints.push(checkpoint);
        }
    }

    history
}

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
    let time_limit_total = 1.8;
    let initial_solution_time_limit = 0.1; // 初期解生成に使う時間

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

    let mut output = best_initial.unwrap_or_else(|| (vec![], 0)).0;

    // 山登り法で改善（変更点以降を再計算）
    let remaining_time = time_limit_total - start_time.elapsed().as_secs_f64();
    if remaining_time > 0.1 && !output.is_empty() {
        let initial_score = evaluate_score(n, k, &adj, &output, &ice_type);
        output = hill_climb(
            n,
            k,
            t,
            &adj,
            &ice_type,
            &reachable_trees,
            output,
            remaining_time,
        );
        let final_score = evaluate_score(n, k, &adj, &output, &ice_type);
        eprintln!("Hill climbing: {} -> {}", initial_score, final_score);
    }

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

// 山登り法（変更点以降を再計算）- スナップショット最適化版
fn hill_climb(
    n: usize,
    k: usize,
    t: usize,
    adj: &[Vec<usize>],
    initial_ice_type: &[char],
    reachable_trees: &[Vec<usize>],
    initial_solution: Vec<i32>,
    time_limit: f64,
) -> Vec<i32> {
    let start_time = Instant::now();
    let mut rng = thread_rng();

    // チェックポイント間隔（100ステップごとにスナップショット）
    let checkpoint_interval = 100;

    // 初期解から履歴を構築
    let mut current_history = build_history_from_solution(
        n, k, adj, initial_ice_type, &initial_solution, checkpoint_interval
    );

    let mut current_solution = initial_solution;
    let mut current_score = evaluate_score(n, k, adj, &current_solution, initial_ice_type);
    let mut best_solution = current_solution.clone();
    let mut best_score = current_score;
    let mut best_history = current_history.checkpoints.clone();

    let mut iterations = 0;
    let mut no_improvement = 0;

    while start_time.elapsed().as_secs_f64() < time_limit {
        iterations += 1;

        // 時間チェック
        if iterations % 100 == 0 {
            if start_time.elapsed().as_secs_f64() >= time_limit {
                break;
            }
        }

        // 近傍解を生成（スナップショットを使用して高速化）
        let neighbor = generate_neighbor_with_recalc(
            n,
            k,
            t,
            adj,
            initial_ice_type,
            reachable_trees,
            &current_solution,
            &current_history,
            &mut rng,
        );

        if neighbor.is_empty() {
            no_improvement += 1;
            if no_improvement > 1000 {
                // 局所最適解に到達した可能性があるので、リスタート
                current_solution = best_solution.clone();
                current_score = best_score;
                // 履歴も再構築
                current_history = build_history_from_solution(
                    n, k, adj, initial_ice_type, &current_solution, checkpoint_interval
                );
                no_improvement = 0;
            }
            continue;
        }

        let neighbor_score = evaluate_score(n, k, adj, &neighbor, initial_ice_type);

        // スコアが改善したら採用
        if neighbor_score > current_score {
            current_solution = neighbor;
            current_score = neighbor_score;
            // 履歴を再構築
            current_history = build_history_from_solution(
                n, k, adj, initial_ice_type, &current_solution, checkpoint_interval
            );
            no_improvement = 0;

            // 最良解を更新
            if current_score > best_score {
                best_solution = current_solution.clone();
                best_score = current_score;
                best_history = current_history.checkpoints.clone();
            }
        } else {
            no_improvement += 1;
            if no_improvement > 1000 {
                // 局所最適解に到達した可能性があるので、リスタート
                current_solution = best_solution.clone();
                current_score = best_score;
                // 履歴も再構築（保存していたチェックポイントを使用）
                current_history = build_history_from_solution(
                    n, k, adj, initial_ice_type, &current_solution, checkpoint_interval
                );
                no_improvement = 0;
            }
        }
    }

    eprintln!(
        "Hill climbing: {} iterations, best score: {}",
        iterations, best_score
    );
    let _ = best_history; // 将来の差分スコアリング用

    best_solution
}

// 近傍解を生成（変更点以降を再計算）- スナップショット最適化版
fn generate_neighbor_with_recalc(
    n: usize,
    k: usize,
    t: usize,
    adj: &[Vec<usize>],
    initial_ice_type: &[char],
    reachable_trees: &[Vec<usize>],
    solution: &[i32],
    history: &StateHistory,
    rng: &mut impl Rng,
) -> Vec<i32> {
    if solution.is_empty() {
        return vec![];
    }

    // 変更位置をランダムに選択（後半を優先）
    let change_pos = if solution.len() > 100 {
        // 後半70%から選択
        let start = (solution.len() as f64 * 0.3) as usize;
        rng.gen_range(start..solution.len())
    } else {
        rng.gen_range(0..solution.len())
    };

    // スナップショットから状態を復元（O(checkpoint_interval)で高速）
    let restored = if change_pos <= history.len() {
        history.restore_to_step(change_pos)
    } else {
        // 履歴が足りない場合は従来通りリプレイ
        let mut state = Checkpoint::new(n, k);
        state.ice_type = initial_ice_type.to_vec();
        for i in 0..change_pos {
            let op = solution[i];
            match op {
                -1 => {
                    if state.pos >= k && state.ice_type[state.pos] == 'W' {
                        state.ice_type[state.pos] = 'R';
                    }
                }
                next_idx => {
                    let next = next_idx as usize;
                    if let Some(prev_pos) = state.prev {
                        if next == prev_pos {
                            return vec![];
                        }
                    }
                    if !adj[state.pos].contains(&next) {
                        return vec![];
                    }
                    state.prev = Some(state.pos);
                    state.pos = next;
                    if state.pos < k {
                        state.shops[state.pos].insert(state.cone.clone());
                        state.cone.clear();
                    } else {
                        state.cone.push(state.ice_type[state.pos]);
                    }
                }
            }
        }
        state
    };

    let mut pos = restored.pos;
    let mut prev = restored.prev;
    let mut ice_type = restored.ice_type;
    let mut cone = restored.cone;
    let mut shops = restored.shops;
    let prefix: Vec<i32> = solution[0..change_pos].to_vec();

    // 変更操作を適用
    let change_type = rng.gen_range(0..3);
    let mut new_prefix = prefix.clone();

    match change_type {
        0 => {
            // 移動先を変更
            if change_pos < solution.len() && solution[change_pos] != -1 {
                let candidates: Vec<usize> = adj[pos]
                    .iter()
                    .filter(|&&next| prev.map(|p| next != p).unwrap_or(true))
                    .copied()
                    .collect();

                if !candidates.is_empty() {
                    let new_next = candidates[rng.gen_range(0..candidates.len())];
                    new_prefix.push(new_next as i32);
                    // 状態を更新
                    prev = Some(pos);
                    pos = new_next;
                    if pos < k {
                        shops[pos].insert(cone.clone());
                        cone.clear();
                    } else {
                        cone.push(ice_type[pos]);
                    }
                } else {
                    return vec![];
                }
            } else {
                return vec![];
            }
        }
        1 => {
            // Flip操作を追加（可能な場合）
            if pos >= k
                && ice_type[pos] == 'W'
                && !cone.is_empty()
                && (0..k).all(|shop_id| shops[shop_id].contains(&cone))
            {
                new_prefix.push(-1);
                ice_type[pos] = 'R';
            } else {
                return vec![];
            }
        }
        2 => {
            // 操作を削除（Flip操作でない場合）
            if change_pos < solution.len() && solution[change_pos] != -1 {
                // 削除するだけ（prefixに追加しない）
            } else {
                return vec![];
            }
        }
        _ => {}
    }

    // 変更点以降を再計算（貪欲に生成）
    let mut suffix = Vec::new();
    let mut current = pos;
    let mut prev_pos = prev;
    let mut ice_type_local = ice_type;
    let mut cone_local = cone;
    let mut shops_local = shops;

    // 残り操作数を計算
    let remaining_ops = t.saturating_sub(new_prefix.len());

    for _step in 0..remaining_ops {
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
        if !cone_local.is_empty() {
            let all_shops_have_it =
                (0..k).all(|shop_id| shops_local[shop_id].contains(&cone_local));

            if all_shops_have_it && current >= k && ice_type_local[current] == 'W' {
                let can_move = adj[current]
                    .iter()
                    .any(|&next| prev_pos.map(|p| next != p).unwrap_or(true));

                if can_move {
                    suffix.push(-1);
                    ice_type_local[current] = 'R';
                    continue;
                }
            }
        }

        // コーンの内容が既にショップに納品されている場合、そのショップを避ける
        let shop_candidates: Vec<usize> = next_candidates
            .iter()
            .filter(|&&v| v < k)
            .filter(|&&shop_id| !shops_local[shop_id].contains(&cone_local))
            .copied()
            .collect();

        if !shop_candidates.is_empty() {
            next_candidates = shop_candidates;
        } else if !cone_local.is_empty() {
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
                let mut scored_trees: Vec<(usize, usize)> = tree_candidates
                    .iter()
                    .map(|&tree| {
                        let score = (0..k)
                            .filter(|&shop_id| {
                                reachable_trees[shop_id].contains(&tree)
                                    && shops_local[shop_id].len() < reachable_trees[shop_id].len()
                            })
                            .count();
                        (tree, score)
                    })
                    .collect();

                scored_trees.sort_by(|a, b| b.1.cmp(&a.1));

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

        suffix.push(next as i32);
        prev_pos = Some(current);
        current = next;

        if current < k {
            shops_local[current].insert(cone_local.clone());
            cone_local.clear();
        } else {
            cone_local.push(ice_type_local[current]);
        }
    }

    // 新しい解を結合
    let mut new_solution = new_prefix;
    new_solution.extend(suffix);

    // 解の妥当性をチェック
    if evaluate_score(n, k, adj, &new_solution, initial_ice_type) == 0 {
        return vec![];
    }

    new_solution
}
