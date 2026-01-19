fn main() {
    proconio::input! {
        d: usize,
        c: [i32; 26],
        s: [[i32; 26]; d],
    }

    let mut last = vec![0; 26];

    fn down_score(last: &[usize], c: &[i32], current_day: i32) -> i32 {
        let mut d_s = 0;
        for i in 0..26 {
            d_s += c[i] * (current_day + 1 - last[i] as i32);
        }
        d_s
    }

    for i in 0..d {
        let current_day = ( i+1 ) as i32;
        let mut change_score = [0; 26];
        for j in 0..26 {
            let before = last[j];
            last[j] = i + 1;
            change_score[j] = s[i][j] - down_score(&last, &c, current_day);
            last[j] = before;
        }
        let idx = change_score.iter().enumerate().max_by_key(|t| t.1).unwrap().0;
        last[idx] = i + 1;
        println!("{}", idx + 1);
    }
}
