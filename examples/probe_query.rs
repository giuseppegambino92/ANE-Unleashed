use ane_unleashed::query::{AneTable, CmpOp, Filter};
fn main() {
    let rows = 20_000_123;
    let a: Vec<f32> = (0..rows).map(|i| ((i * 7919) % 1000) as f32 - 500.0).collect();
    let cols = [a.as_slice(), a.as_slice(), a.as_slice()];
    let t = AneTable::from_f32(&cols).unwrap();
    let f = [Filter { col: 0, op: CmpOp::Gt, value: 50.0 }];
    for (sum, mask) in [(None, false), (Some(2), false), (None, true), (Some(2), true)] {
        match t.query(&f, sum, mask) {
            Ok(r) => println!("sum={sum:?} mask={mask}: OK count={}", r.count),
            Err(e) => println!("sum={sum:?} mask={mask}: ERR {}", &e.to_string()[..140]),
        }
    }
}
