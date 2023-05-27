mod perceptron;
mod modelelineaire;

use perceptron::MyMLP;
use modelelineaire::LinRegressor;
use rusty_machine::linalg::Matrix;
use rusty_machine::linalg::Vector;

fn main() {
    let mut mlp = MyMLP::new(vec![2, 0, 1]);
    mlp._propagate(&[0.0, 0.0], false, MyMLP::sigmoid);
    println!("{:?}", mlp.X);

    let mut test_1_all_samples_inputs: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0]
    ];

    let test_1_all_samples_expected_outputs: Vec<Vec<f64>> = vec![
        vec![1.0],
        vec![0.0],
        vec![0.0]
    ];
/*
    // Afficher les résultats avant l'entraînement
    println!("Résultats avant l'entraînement :");
    for inputs in &test_1_all_samples_inputs {
        let predictions = mlp.predict(inputs, false, MyMLP::sigmoid);
        println!("Entrées : {:?} | Prédictions : {:?}", inputs, predictions);
    }
*/
    // Train du perceptron
    let layer_sizes = vec![3, 4, 1];
    let mut train = MyMLP::new(layer_sizes);
    let all_samples_inputs: Vec<Vec<f64>> = test_1_all_samples_inputs.clone();
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
        MyMLP::sigmoid,
    );
/*
    // Afficher les résultats après l'entraînement du perceptron
    println!("Résultats après l'entraînement du perceptron :");
    for inputs in &test_1_all_samples_inputs {
        let predictions = mlp.predict(&inputs, is_classification, MyMLP::sigmoid);
        println!("Entrées : {:?} | Prédictions : {:?}", inputs, predictions);
    }
*/
    // Entraînement du modèle linéaire
    let mut lin_mod = LinRegressor::default();
    lin_mod.train(
        &Matrix::new(all_samples_inputs.len(), all_samples_inputs[0].len(), all_samples_inputs.concat()),
        &Vector::new(all_samples_expected_outputs.concat()),
    );

    // Afficher les résultats du perceptron et du modèle linéaire
    println!("Résultats avec le perceptron et le modèle linéaire :");
    for inputs in &test_1_all_samples_inputs {
        let predictions_mlp = mlp.predict(&inputs, is_classification, MyMLP::sigmoid);
        let predictions_lin = lin_mod.predict(&Matrix::new(1, inputs.len(), inputs.clone()));

        println!("Entrées : {:?} | Prédictions MLP : {:?} | Prédictions modèle linéaire : {:?}", inputs, predictions_mlp, predictions_lin);
    }
}