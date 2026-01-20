use rand::Rng;
use std::time::Instant;

struct Problem {
    d: usize,
    c: Vec<i32>,
    s: Vec<Vec<i32>>
}

fn calc_score(problem: &Problem, schedule: &[usize]) -> i32 {
    let mut total_score = 0;
    let mut last = vec![0usize; 26];

    for i in 0..problem.d {
        let selected_contest_idx = schedule[i];
        total_score += problem.s[i][selected_contest_idx];
        for j in 0..26 {
            total_score -= problem.c[j] * (i + 1 - last[j]) as i32;
        }
        last[selected_contest_idx] = i + 1;
    }
    total_score
}

fn main() {
    proconio::input! {
        d: usize,
        c: [i32; 26],
        s: [[i32; 26]; d],
    }

    let start = Instant::now();
    let time_limit = 1.0;
    let start_temp = 500.0;
    let end_temp = 1.0;
    let problem = Problem{ d, c, s };
    let mut rng = rand::thread_rng();
    let mut last = vec![0; 26];
    let mut init_schedule = vec![0; d];

    for i in 0..d {
        let current_day = ( i+1 ) as i32;
        let mut change_score = [0; 26];
        for j in 0..26 {
            let before = last[j];
            last[j] = i + 1;
            change_score[j] = problem.s[i][j] + &problem.c[j] * (d - before) as i32;
            last[j] = before;
        }
        let idx = change_score.iter().enumerate().max_by_key(|t| t.1).unwrap().0;
        last[idx] = i + 1;
        init_schedule[i] = idx;
    }

    let mut schedule = init_schedule;
    let mut score = calc_score(&problem, &schedule);

    loop {
        if start.elapsed().as_secs_f64() > time_limit {
            break;
        }

        let elapsed = start.elapsed().as_secs_f64();
        let temp = start_temp + (end_temp - start_temp) * elapsed / time_limit;

        let day = rng.gen_range(0..d);
        let contest = rng.gen_range(0..26);

        let old_contest = schedule[day];
        schedule[day] = contest;

        let new_score = calc_score(&problem, &schedule);
        let diff = new_score - score;

        if diff > 0 || rng.r#gen::<f64>() < ((diff as f64 / temp).exp()) {
            score = new_score;
        } else {
            schedule[day] = old_contest;
        }
    }

    for i in 0..d {
        println!("{}", schedule[i] + 1);
    }
}
