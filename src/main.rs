use ndarray::{Array1, ArrayView1};

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
fn forwardback(
    x: ArrayView1<f32>,
    y: ArrayView1<f32>,
    theta: ArrayView1<f32>,
) -> (f32, Array1<f32>) {
    let denom = 1. / (x.len() as f32);
    let mean = |val: Array1<f32>| val.sum() * denom;

    let yhat = forward(x, theta);
    let dldyhat = yhat - y;
    let loss = mean(dldyhat.mapv(|x| x * x)) * 0.5;

    let mut gradient = Array1::zeros(theta.raw_dim());
    let mut tmp_x = Array1::<f32>::ones(x.raw_dim());

    for i in 0..gradient.len() {
        gradient[i] = mean(dldyhat.clone() * tmp_x.clone());
        tmp_x = tmp_x * x;
    }

    return (loss, gradient);
}

fn main() {
    let param = Array1::from_vec(vec![3.4, 2.9, 4.5]);
    let param2 = Array1::from_vec(vec![3., 3., 5.]);

    let x = Array1::from_vec(vec![1., 2., 3., 4., 5., 6., 7., 8.]);
    let y = forward(x.view(), param.view());
    let yhat = forward(x.view(), param2.view());
    let (loss, gradient) = forwardback(x.view(), y.view(), param2.view());

    println!("Y: {}", y);
    println!("Yhat: {}", yhat);

    println!("Loss: {}", loss);
    println!("Gradient: {}", gradient);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward() {
        let param = Array1::from_vec(vec![3.4, 2.9, 4.5]);
        let arr = Array1::from_vec(vec![1., 2., 3., 4., 5., 6., 7., 8.]);
        let result = forward(arr.view(), param.view());
        assert_eq!(
            result.to_vec(),
            vec![10.8, 27.2, 52.6, 87., 130.4, 182.8, 244.2, 314.6]
        );
    }

    #[test]
    fn test_backprop() {
        let param = Array1::from_vec(vec![3.4, 2.9, 4.5]);
        let param2 = Array1::from_vec(vec![3., 3., 5.]);

        let x = Array1::from_vec(vec![1., 2., 3., 4., 5., 6., 7., 8.]);
        let y = forward(x.view(), param.view());
        let (loss, gradient) = forwardback(x.view(), y.view(), param2.view());
        assert_eq!(
            y.to_vec(),
            vec![10.8, 27.2, 52.6, 87., 130.4, 182.8, 244.2, 314.6]
        );
        assert_eq!(loss, 140.09);
        assert_eq!(gradient.to_vec(), vec![12.8, 81.75, 554.25]);
    }
}
