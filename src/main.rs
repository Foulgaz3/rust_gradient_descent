use rand::prelude::*;
use ndarray::{Array, Array1, ArrayView1};
use std::env;

fn forward(x: ArrayView1<f32>, theta: ArrayView1<f32>) -> Array1<f32> {
    let mut result = Array1::from_elem(x.raw_dim(), theta[0]);
    let mut tmp_x = x.to_owned();
    for i in 1..theta.len() {
        let b: f32 = theta[i];
        result = result + b * tmp_x.clone();
        tmp_x = tmp_x * x;
    }
    return result;
}

/// Performs forward and back propagation
/// returns (loss, gradient)
/// 
/// TODO: blend forward and backward computation
fn forwardback(x: ArrayView1<f32>, y: ArrayView1<f32>, theta: ArrayView1<f32>) -> (f32, Array1<f32>) {
    let yhat = forward(x, theta);
    let dLdyhat = yhat - y;
    let pN = 1./(x.len() as f32);
    let mean = |val: Array1<f32>| {val.sum() * pN};
    let loss = mean(dLdyhat.clone()*dLdyhat.view())*0.5;
    let mut gradient = Array1::zeros(theta.raw_dim());
    // let mut tmp_x = Array1::<f32>::ones(theta.raw_dim());
    // for i in 0..gradient.len() {
    //     gradient[i] = mean(dLdyhat.clone()*tmp_x.clone());
    //     tmp_x = tmp_x * x
    // }
    return (loss, gradient)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward() {
        let param = Array1::from_vec(vec![3.4, 2.9, 4.5]);
        let arr = Array1::from_vec(vec![1., 2., 3., 4., 5., 6., 7., 8.]);
        let result = forward(arr.view(), param.view());
        assert_eq!(result.to_vec(), vec![10.8, 27.2, 52.6, 87., 130.4, 182.8, 244.2, 314.6]);
    }
}

fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    let param = Array1::from_vec(vec![3.4, 2.9, 4.5]);
    let param2 = Array1::from_vec(vec![3., 3., 5.]);

    let X = Array1::from_vec(vec![1., 2., 3., 4., 5., 6., 7., 8.]);
    let Y = forward(X.view(), param.view());
    let Yhat = forward(X.view(), param2.view());
    let (loss, gradient) = forwardback(X.view(), Y.view(), param2.view());

    println!("Y: {}", Y);
    println!("Yhat: {}", Yhat);
    
    println!("Loss: {}", loss);
}
