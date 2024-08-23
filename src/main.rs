use ndarray::{Array, Array1, Dimension};

struct Adam<D: Dimension> {
    lr: f32,
    b1: f32,
    b2: f32,
    t: i32,
    m: Array<f32, D>,
    v: Array<f32, D>,
    epsilon: f32,
}

impl<D: Dimension> Adam<D> {
    fn new(param: &Array<f32, D>, lr: f32, b1: f32, b2: f32) -> Self {
        let shape = param.raw_dim();
        Adam {
            lr,
            b1,
            b2,
            t: 0,
            m: Array::<f32, D>::zeros(shape.clone()), // Initialize m to zeros
            v: Array::<f32, D>::zeros(shape.clone()), // Initialize v to zeros
            epsilon: 1e-8,
        }
    }

    fn update(&mut self, gradient: &Array<f32, D>) -> Array<f32, D> {
        self.t += 1;
        self.m = &self.m * self.b1 + gradient * (1.0 - self.b1);
        self.v = &self.v * self.b2 + gradient.mapv(|g| g.powi(2)) * (1.0 - self.b2);
        let m_hat = &self.m / (1.0 - self.b1.powi(self.t));
        let v_hat = &self.v / (1.0 - self.b2.powi(self.t));
        m_hat / (v_hat.mapv(f32::sqrt) + self.epsilon) * self.lr
    }
}

fn forward(x: &Array1<f32>, theta: &Array1<f32>) -> Array1<f32> {
    let mut result = Array1::from_elem(x.raw_dim(), theta[0]);
    let mut tmp_x = x.to_owned();
    for i in 1..theta.len() {
        let b: f32 = theta[i];
        result = result + b * tmp_x.clone();
        tmp_x = tmp_x * x;
    }
    result
}

/// Performs forward and back propagation
/// returns (loss, gradient)
fn forwardback(x: &Array1<f32>, y: &Array1<f32>, theta: &Array1<f32>) -> (f32, Array1<f32>) {
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

    (loss, gradient)
}

fn main() {
    let param = Array1::from_vec(vec![3.4, 2.9, 4.5]);
    let mut param2 = Array1::from_vec(vec![3., 3., 5.]);

    let mut adam = Adam::new(&param2.clone(), 0.01, 0.9, 0.999);

    // let x = Array1::from_vec(vec![1., 2., 3., 4., 5., 6., 7., 8.]);
    let x = Array1::linspace(-2.5, 2.5, 50);
    let y = forward(&x, &param);

    for i in 0..1000 {
        let (loss, gradient) = forwardback(&x.clone(), &y.clone(), &param2);
        param2 = &param2 - &adam.update(&gradient);
        if i % 100 == 0 {
            println!("round: {i}, loss: {loss}")
        }
    }

    println!("Final Parameters: {param2}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward() {
        let param = Array1::from_vec(vec![3.4, 2.9, 4.5]);
        let arr = Array1::from_vec(vec![1., 2., 3., 4., 5., 6., 7., 8.]);
        let result = forward(&arr, &param);
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
        let y = forward(&x, &param);
        let (loss, gradient) = forwardback(&x, &y, &param2);
        assert_eq!(
            y.to_vec(),
            vec![10.8, 27.2, 52.6, 87., 130.4, 182.8, 244.2, 314.6]
        );
        assert_eq!(loss, 140.09);
        assert_eq!(gradient.to_vec(), vec![12.8, 81.75, 554.25]);
    }
}
