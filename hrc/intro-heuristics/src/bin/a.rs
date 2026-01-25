use rand::Rng;
use std::time::Instant;

struct Problem {
    d: usize,
    c: Vec<i32>,
    s: Vec<Vec<i32>>,
}

impl Problem {
    fn calc_score(&self, schedule: &[usize]) -> i32 {
        let mut total_score = 0;
        let mut last = vec![0usize; 26];

        for i in 0..self.d {
            let selected = schedule[i];
            last[selected] = i + 1;
            total_score += self.s[i][selected];
            for j in 0..26 {
                total_score -= self.c[j] * (i + 1 - last[j]) as i32;
            }
        }
        total_score
    }

    fn diff_score(&self, state: &State, day: usize, new_contest: usize) -> i32 {
        let day1 = day + 1;
        let old_contest = state.schedule[day];

        let (prev_old, next_old) = state.find_neighbors(old_contest, day, self.d, true);
        let (prev_new, next_new) = state.find_neighbors(new_contest, day, self.d, false);

        self.s[day][new_contest]
            - self.s[day][old_contest]
            - self.c[old_contest] * (day1 - prev_old) as i32 * (next_old - day1) as i32
            + self.c[new_contest] * (day1 - prev_new) as i32 * (next_new - day1) as i32
    }
}

struct State {
    schedule: Vec<usize>,
    occurrences: Vec<Vec<usize>>,
    current_score: i32,
}

impl State {
    fn new(problem: &Problem, schedule: Vec<usize>) -> Self {
        let mut occurrences = vec![vec![]; 26];
        for (day, &t) in schedule.iter().enumerate() {
            occurrences[t].push(day + 1);
        }
        for occ in &mut occurrences {
            occ.push(problem.d + 1);
        }
        let current_score = problem.calc_score(&schedule);
        State {
            schedule,
            occurrences,
            current_score,
        }
    }

    fn find_neighbors(
        &self,
        contest: usize,
        day: usize,
        d: usize,
        is_current: bool,
    ) -> (usize, usize) {
        let day1 = day + 1;
        let occ = &self.occurrences[contest];
        let pos = occ.partition_point(|&x| x < day1);
        let prev = if pos > 0 { occ[pos - 1] } else { 0 };
        let next_pos = if is_current { pos + 1 } else { pos };
        let next = if next_pos < occ.len() {
            occ[next_pos]
        } else {
            d + 1
        };
        (prev, next)
    }

    fn update(&mut self, new_contest: usize, day: usize, diff: i32) {
        let day1 = day + 1;
        self.current_score += diff;
        let old_contest = self.schedule[day];

        if let Ok(pos) = self.occurrences[old_contest].binary_search(&day1) {
            self.occurrences[old_contest].remove(pos);
        }
        let pos = self.occurrences[new_contest].partition_point(|&x| x < day1);
        self.occurrences[new_contest].insert(pos, day1);
        self.schedule[day] = new_contest;
    }
}

fn main() {
    proconio::input! {
        d: usize,
        c: [i32; 26],
        s: [[i32; 26]; d],
    }

    let start = Instant::now();
    let time_limit = 1.9;
    let start_temp = 2000.0;
    let end_temp = 10.0;
    let problem = Problem { d, c, s };
    let mut rng = rand::thread_rng();

    let mut init_schedule = vec![0; d];
    let mut last = vec![0i32; 26];
    for i in 0..d {
        let (best_type, _) = (0..26)
            .map(|j| (j, problem.s[i][j] + problem.c[j] * (i as i32 + 1 - last[j])))
            .max_by_key(|&(_, v)| v)
            .unwrap();
        last[best_type] = i as i32 + 1;
        init_schedule[i] = best_type;
    }

    let mut state = State::new(&problem, init_schedule);
    let mut best_schedule = state.schedule.clone();
    let mut best_score = state.current_score;
    let mut loop_count = 0u64;

    loop {
        if loop_count % 100 == 0 && start.elapsed().as_secs_f64() > time_limit {
            break;
        }
        loop_count += 1;

        let temp =
            start_temp + (end_temp - start_temp) * start.elapsed().as_secs_f64() / time_limit;
        let day = rng.gen_range(0..d);
        let old = state.schedule[day];
        let r = rng.gen_range(0..25);
        let new = if r >= old { r + 1 } else { r };

        let diff = problem.diff_score(&state, day, new);

        if diff > 0 || rng.r#gen::<f64>() < ((diff as f64 / temp).exp()) {
            state.update(new, day, diff);
            if state.current_score > best_score {
                best_score = state.current_score;
                best_schedule = state.schedule.clone();
            }
        }
    }

    for t in best_schedule {
        println!("{}", t + 1);
    }
}
