mod perceptron;
use perceptron::{MyMLP};


fn main() {
    let mut mlp = MyMLP::new(vec![2, 0, 1]);
    mlp._propagate(&[0.0, 0.0], false);
    println!("{:?}", mlp.X);




    let mut test_1_all_samples_inputs: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0]
    ];


    let test_1_all_samples_expected_outputs: Vec<Vec<f64>> = vec![
        vec![1.0],
        vec![-1.0],
        vec![-1.0]
    ];


    // Afficher les résultats avant l'entraînement
    println!("Résultats avant l'entraînement :");
    for inputs in &test_1_all_samples_inputs {
        let predictions = mlp.predict(inputs, false);
        println!("Entrées : {:?} | Prédictions : {:?}", inputs, predictions);
    }

    //train
    let layer_sizes = vec![2, 0, 1];
    let mut train = MyMLP::new(layer_sizes);
    let all_samples_inputs: Vec<Vec<f64>> = test_1_all_samples_inputs;
    let all_samples_expected_outputs: Vec<Vec<f64>> = test_1_all_samples_expected_outputs;
    let is_classification = true;
    let iteration_count = 100000;
    let alpha = 0.01;

    mlp.train(
        &all_samples_inputs,
        &all_samples_expected_outputs,
        is_classification,
        iteration_count,
        alpha,
    );

    // Afficher les résultats après l'entraînement
    println!("Résultats après l'entraînement :");
    for inputs in &test_1_all_samples_inputs {
        let predictions = mlp.predict(inputs, true);
        println!("Entrées : {:?} | Prédictions : {:?}", inputs, predictions);
    }









}
