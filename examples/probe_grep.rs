use ane_unleashed::grep::AneGrep;
use std::time::Instant;
fn main() {
    let a: Vec<usize> = std::env::args().skip(1).map(|s| s.parse().unwrap()).collect();
    let (np, proj, rows) = (a[0], a[1], a[2]);
    let pats: Vec<Vec<u8>> = (0..np).map(|i| format!("pattern{i:05}xyz").into_bytes()).collect();
    let p: Vec<&[u8]> = pats.iter().map(|v| v.as_slice()).collect();
    let t = Instant::now();
    match AneGrep::new(&p, proj, rows) {
        Ok(g) => {
            let ct = t.elapsed().as_secs_f64() * 1e3;
            let seg = g.segments_per_dispatch();
            let text = vec![b'a'; seg * 16384 * 4];
            let _ = g.scan(&text).unwrap();
            let (_, st) = g.scan(&text).unwrap();
            println!("np={np} proj={proj} segs={seg}: compile {ct:.0} ms, ANE {:.2} ms/dispatch => ANE-only {:.2} GB/s, end-to-end {:.2} GB/s (copy {:.1} ms confirm {:.1} ms)",
                st.ane_s * 1e3 / 4.0, text.len() as f64 / st.ane_s / 1e9, text.len() as f64 / st.total_s / 1e9, st.copy_s * 1e3, st.confirm_s * 1e3);
        }
        Err(e) => println!("np={np} proj={proj} rows={rows}: FAIL {}", &e.to_string()[..160.min(e.to_string().len())]),
    }
}
