use image::{DynamicImage, GenericImageView};
use ndarray::prelude::*;
use std::path::Path;

fn load_and_preprocess_image(image_path: &str) -> Vec<f32> {
    // Charger une image à partir d'un fichier
    // let img = image::open("image.png").unwrap();
    let img = image::open(&Path::new(image_path)).unwrap();

    // Convertir l'image en tableau ndarray
    let mut img_data: Vec<u8> = img.raw_pixels();
    let img_ndarray = Array::from_shape_vec((img.height() as usize, img.width() as usize, img.color().channel_count() as usize), img_data).unwrap();

    // Réduire la taille de l'image
    let resized_img_ndarray = imageops::resize(&img_ndarray, 100, 100, imageops::FilterType::Triangle);

    // Normaliser les valeurs des pixels
    let normalized_img_ndarray = (resized_img_ndarray.mapv(|x| x as f32 / 255.0) - 0.5) * 2.0;

    // Convertir le tableau ndarray en vecteur
    let mut normalized_img_data: Vec<f32> = Vec::new();
    normalized_img_ndarray.iter().for_each(|&x| normalized_img_data.push(x));

    // Utiliser le vecteur pour entraîner un modèle de machine learning
    // ...
    return normalized_img_data;
}

fn put_image_in_array() {
    // Créer un vecteur de deux lignes où chaque élément est un vecteur
    let mut matrix: Vec<f32> = Vec::new();
    for conor in 0..30 {
        normalized_img_data.push(conor);
    }
    for tyson in 0..30 {
        normalized_img_data.push(tyson);
    }

    let mut matrix: Vec<Vec<T>> = Vec::new(); 
    for i in 1..31 {
        let mut link: String = "./dataset/conor_mcgregor/";
        link.push(i.to_string());
        matrix.push(load_and_preprocess_image(link))
    }
    matrix.transpose();
    let mut vector: Vec<Vec<T>> = Vec::new(); 
    for i in 1..29 {
        let mut link: String = "./dataset/mike_tyson/";
        link.push(i.to_string());
        vector.push(load_and_preprocess_image(link))
    }
    vector.transpose();
    matrix.push(vector[0]);

    //PROBLEME RESOLU !!!!!!!
    //But : On ne veut pas seulement 3 éléments par vecteurs mais on en veut 30. Pour cela
    //le mieux est de faire une boucle for qui ajoute chaque élément un par un. (avec push)
    //-> Problème : après avoir rempli le 1er vec -> les éléments s'affichent en colonnes et pas en lignes
    //Donc après le 1 er vec, on utilise transpose pour que le 1er vec corresponde à la 1re ligne
    //On effectue la même opération pour le 2ème vec

    // Afficher le contenu du vecteur
    println!("{:?}", matrix);
}
//1 et -1 sur photo -> tous les labels (Y_train)
//X_train 

//0 1 2 3 4 5
//6 7 8 9 0 0