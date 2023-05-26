use rand::Rng;

pub(crate) struct MyMLP {
    d: Vec<usize>,
    L: usize,
    W: Vec<Vec<Vec<f64>>>,
    pub(crate) X: Vec<Vec<f64>>,
    deltas: Vec<Vec<f64>>,
}

impl MyMLP {
    pub(crate) fn new(npl: Vec<usize>) -> Self {
        let d = npl.clone();
        let L = npl.len() - 1;
        let mut W = Vec::new();

        for l in 0..=L {
            W.push(Vec::new());

            if l == 0 {
                continue;
            }

            for _i in 0..=npl[l - 1] {
                W[l].push(Vec::new());
                for j in 0..=npl[l] {
                    W[l][_i].push(if j == 0 { 0.0 } else { rand::thread_rng().gen_range(-1.0..=1.0) });
                }
            }
        }

        let mut X = Vec::new();
        for l in 0..=L {
            X.push(Vec::new());
            for j in 0..=npl[l] {
                X[l].push(if j == 0 { 1.0 } else { 0.0 });
            }
        }

        let mut deltas = Vec::new();
        for l in 0..=L {
            deltas.push(Vec::new());
            for _j in 0..=npl[l] {
                deltas[l].push(0.0);
            }
        }

        MyMLP {
            d,
            L,
            W,
            X,
            deltas,
        }
    }

    pub(crate) fn _propagate(&mut self, inputs: &[f64], is_classification: bool) {
        for j in 0..self.d[0] {
            self.X[0][j + 1] = inputs[j];
        }

        for l in 1..=self.L {
            for j in 1..=self.d[l] {
                let mut total = 0.0;
                for i in 0..=self.d[l - 1] {
                    total += self.W[l][i][j] * self.X[l - 1][i];
                }

                if l < self.L || is_classification {
                    total = total.tanh();
                }

                self.X[l][j] = total;
            }
        }
    }

    pub(crate) fn predict(&mut self, inputs: &[f64], is_classification: bool) -> Vec<f64> {
        self._propagate(inputs, is_classification);
        self.X[self.L][1..].to_vec()
    }


    pub(crate) fn train(
        &mut self,
        all_samples_inputs: &[Vec<f64>],
        all_samples_expected_outputs: &[Vec<f64>],
        is_classification: bool,
        iteration_count: usize,
        alpha: f64,
    ) {
        for _ in 0..iteration_count {
            let k = rand::thread_rng().gen_range(0..all_samples_inputs.len());
            let inputs_k = &all_samples_inputs[k];
            let y_k = &all_samples_expected_outputs[k];

            self._propagate(inputs_k, is_classification);

            for j in 1..=self.d[self.L] {
                self.deltas[self.L][j] = self.X[self.L][j] - y_k[j - 1];
                if is_classification {
                    self.deltas[self.L][j] *= 1.0 - self.X[self.L][j].powi(2);
                }
            }

            for l in (1..=self.L).rev() {
                for i in 1..=self.d[l - 1] {
                    let mut total = 0.0;
                    for j in 1..=self.d[l] {
                        total += self.W[l][i][j] * self.deltas[l][j];
                    }
                    self.deltas[l - 1][i] = (1.0 - self.X[l - 1][i].powi(2)) * total;
                }
            }

            for l in 1..=self.L {
                for i in 0..=self.d[l - 1] {
                    for j in 1..=self.d[l] {
                        self.W[l][i][j] -= alpha * self.X[l - 1][i] * self.deltas[l][j];
                    }
                }
            }
        }
    }
}
