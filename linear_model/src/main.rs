use rand::Rng;

struct Perceptron {
    weights: Vec<f64>,
}

impl Perceptron {
    fn new(num_features: usize) -> Perceptron {
        let weights = vec![0.0; num_features + 1];
        Perceptron { weights }
    }

    fn _activation_function(&self, value: f64) -> f64 {
        if value >= 0.0 {
            1.0
        } else {
            0.0
        }
    }

    fn train(&mut self, inputs: &[Vec<f64>], targets: &[f64], num_iterations: usize) {
        let learning_rate = 0.01;
        let mut rng = rand::thread_rng();

        for _ in 0..num_iterations {
            let k = rng.gen_range(0..inputs.len());
            let yk = targets[k];
            let mut xk = inputs[k].clone();
            xk.insert(0, 1.0);

            let _signal: f64 = self.weights.iter().zip(xk.iter()).map(|(&w, &x)| w * x).sum();

            let g_xk = if self.weights.iter().zip(xk.iter()).map(|(&w, &x)| w * x).sum::<f64>() >= 0.0 {
                1.0
            } else {
                0.0
            };

            for (w, &x) in self.weights.iter_mut().zip(xk.iter()) {
                *w += learning_rate * (yk - g_xk) * x;
            }
        }
    }
}

fn main() {
    let inputs: Vec<Vec<f64>> = vec![
        vec![0.2, 0.3],
        vec![0.4, 0.1],
        vec![0.6, 0.7],
    ];
    let targets: Vec<f64> = vec![1.0, 0.0, 1.0];

    let mut perceptron = Perceptron::new(2);
    perceptron.train(&inputs, &targets, 10);

    let test_inputs: Vec<Vec<f64>> = vec![
        vec![0.1, 0.4],
        vec![0.3, 0.2],
        vec![0.5, 0.6],
    ];
    let test_targets: Vec<f64> = vec![1.0, 0.0, 1.0];

    let mut correct_predictions = 0;

    for i in 0..test_inputs.len() {
        let prediction =
            test_inputs[i][0] * perceptron.weights[1] + test_inputs[i][1] * perceptron.weights[2] + perceptron.weights[0];
        let target = test_targets[i];

        if (prediction >= 0.0 && target == 1.0) || (prediction < 0.0 && target == 0.0) {
            correct_predictions += 1;
        }
    }

    let accuracy = correct_predictions as f64 / test_inputs.len() as f64;
    println!("Accuracy: {:.2}", accuracy);
}

