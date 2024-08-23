use rand::prelude::*;

type param = (f64, f64, f64);
fn main() {
    let mut rng = thread_rng();
    let mut theta: param = rng.gen();
    println!("theta: {:?}", theta);
    println!("Hello, world!");
}
