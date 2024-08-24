# Rust Gradient Descent

Quick rust project I wrote to teach myself how to use the [ndarray][ndarray] crate

Implements gradient descent and an Adam optimizer to tune a polynomial to a list of points.

[ndarray]: https://docs.rs/ndarray/latest/ndarray/

## Training Log

```plaintext
Ideal Parameters  : [3.4, 2.9, 4.5]
Initial Parameters: [3.0, 3.0, 5.0]

round:   0, loss: 0.714512705803
round: 100, loss: 0.018291808665
round: 200, loss: 0.000300833053
round: 300, loss: 0.000000650363
round: 400, loss: 0.000000000072
round: 500, loss: 0.000000000011
round: 600, loss: 0.000000000007
round: 700, loss: 0.000000000007
round: 800, loss: 0.000000000007
round: 900, loss: 0.000000000007

Final Parameters: [3.3999956, 2.9, 4.500002]
```
