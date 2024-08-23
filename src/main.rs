use rand::prelude::*;
use ndarray::{Array, Array1, ArrayView1};

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


type param = (f64, f64, f64);

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
    let param = Array1::from_vec(vec![3.4, 2.9, 4.5]);
    let arr = Array1::from_vec(vec![1., 2., 3., 4., 5., 6., 7., 8.]);
    let modified = forward(arr.view(), param.view());
    println!("Array: {}", arr);
    println!("Modified: {}", modified);
}
