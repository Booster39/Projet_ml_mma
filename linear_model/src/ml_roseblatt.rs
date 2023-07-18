use rand::Rng;

#[repr(C)]
pub struct Perceptron {
    weights: Vec<f64>,
}

#[no_mangle]
pub extern "C" fn new_perceptron(num_features: usize) -> *mut Perceptron {
    let weights = vec![0.0; num_features + 1];
    Box::into_raw(Box::new(Perceptron { weights }))
}

#[no_mangle]
pub extern "C" fn train(
    perceptron: *mut Perceptron,
    inputs: *const *const f64,
    targets: *const f64,
    num_samples: usize,
    num_iterations: usize,
) {
    let perceptron = unsafe { &mut *perceptron };
    let inputs = unsafe { std::slice::from_raw_parts(inputs, num_samples) };
    let targets = unsafe { std::slice::from_raw_parts(targets, num_samples) };

    let learning_rate = 0.01;
    let mut rng = rand::thread_rng();

    for _ in 0..num_iterations {
        let k = rng.gen_range(0..num_samples);
        let yk = targets[k];
        let input_slice = unsafe { std::slice::from_raw_parts(inputs[k], perceptron.weights.len()) };
        let mut xk = Vec::with_capacity(perceptron.weights.len() + 1);
        xk.push(1.0); // Ajoute le biais en tant que premier élément
        xk.extend_from_slice(input_slice); // Copie les valeurs du pointeur brut dans xk

        let mut sum = 0.0;
        for i in 0..perceptron.weights.len() {
            sum += perceptron.weights[i] * xk[i];
        }
        let g_xk = if sum >= 0.0 { 1.0 } else { 0.0 };

        for i in 0..perceptron.weights.len() {
            perceptron.weights[i] += learning_rate * (yk - g_xk) * xk[i];
        }
    }
}

#[no_mangle]
pub extern "C" fn predict(perceptron: *const Perceptron, input: *const f64) -> f64 {
    let perceptron = unsafe { &*perceptron };
    let input_slice = unsafe { std::slice::from_raw_parts(input, perceptron.weights.len()) };

    let mut sum = 0.0;
    for i in 0..perceptron.weights.len() {
        sum += perceptron.weights[i] * input_slice[i];
    }

    if sum >= 0.0 {
        1.0
    } else {
        0.0
    }
}

#[no_mangle]
pub extern "C" fn free_perceptron(perceptron: *mut Perceptron) {
    if !perceptron.is_null() {
        unsafe { Box::from_raw(perceptron) };
    }
}
