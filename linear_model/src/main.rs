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
    vec![0.6, 0.7]];

    let targets: Vec<f64> = vec![
        1.0,
        0.0,
        1.0];


    let mut perceptron = Perceptron::new(2);
    perceptron.train(&inputs, &targets, 10);

    let mut predicted_labels = Vec::new();
    let mut predicted_x1 = Vec::new();
    let mut predicted_x2 = Vec::new();

    for x1 in 0..100 {
        for x2 in 0..100 {
            let x1_normalized = x1 as f64 / 100.0;
            let x2_normalized = x2 as f64 / 100.0;

            predicted_x1.push(x1_normalized);
            predicted_x2.push(x2_normalized);

            let prediction = x1_normalized * perceptron.weights[1] + x2_normalized * perceptron.weights[2] + perceptron.weights[0];
            let label = if prediction >= 0.0 { "pink" } else { "lightskyblue" };

            predicted_labels.push(label);
        }
    }
    for i in 0..predicted_x1.len() {
        println!(
            "x1: {:.2}, x2: {:.2}, label: {}",
            predicted_x1[i], predicted_x2[i], predicted_labels[i]
        );
    }
}
