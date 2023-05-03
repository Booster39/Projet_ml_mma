//importation de slice random pour le mélange des données
use rand::seq::SliceRandom;
//Utiliser un HashSet pour ne pas avoir de doublons
use std::collections::HashSet;

//Fonction de séparation des données en données d'entrainement et de test
//en parametre vecteur de tuples (voir la doc de rust pour les tuples)
fn data_separation(data: Vec<(Vec<f32>, i32)>) -> (Vec<(Vec<f32>, i32)>, Vec<(Vec<f32>, i32)>) {

    //déclaration des vecteurs d'entrainement et de test
    let mut train_data= Vec::new();
    let mut test_data= Vec::new();

    //déclaration du HashSet pour ne pas avoir de doublons
    let mut label_data = HashSet::new();

    for(_, label) in &data {
        label_data.insert(label);
    }


    for label in label_data.iter() {
        let mut data_label:Vec<(Vec<f32>, i32)> = data.iter();
    }


}