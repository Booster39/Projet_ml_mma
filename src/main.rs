mod perceptron;
mod modele_lineaire;
mod process;

use perceptron::MyMLP;
use modele_lineaire::LinRegressor;
use rusty_machine::linalg::{Matrix, Vector};
use process::*;
use rand::Rng;

fn main() {

    // Test avec une image
    put_image_in_array();
    println!();
    println!();
    println!();


    // Training of the model
    let mut mlp = MyMLP::new(vec![2, 0, 1]);
    mlp._propagate(&[0.0, 0.0], false, MyMLP::tanh);
    println!("{:?}", mlp.X);

    let mut test_1_all_samples_inputs: Vec<Vec<f64>> = Vec::new();
    let input_count = 1;  // Nombre d'échantillons d'entrée
    let input_base = -1;  // Valeur minimale des échantillons d'entrée

    // Générer des nombres aléatoires pour chaque échantillon
    for _ in input_base..input_count {
        let input: Vec<f64> = (0..2).map(|_| rand::thread_rng().gen_range(0.0..1.0)).collect();
        test_1_all_samples_inputs.push(input);
    }


    let test_1_all_samples_expected_outputs: Vec<Vec<f64>> = vec![
        vec![1.0],
        vec![0.0],
        vec![-1.0]
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
        MyMLP::tanh,
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
        let predictions_mlp = mlp.predict(&inputs, is_classification, MyMLP::tanh);
        let predictions_lin = lin_mod.predict(&Matrix::new(1, inputs.len(), inputs.clone()));

        println!("Entrées : {:?} | Prédictions MLP : {:?} | Prédictions modèle linéaire : {:?}", inputs, predictions_mlp, predictions_lin);
    }
}