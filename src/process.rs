use image::{DynamicImage, GenericImageView};
use ndarray::prelude::*;
use std::path::Path;

fn load_and_preprocess_image(image_path: &str, output_size: (usize, usize)) -> Array3<f32> {
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
    matrix.push(vec![p1, p2, p3]);
    matrix.push(vec![p4, p5, p6]);

    // Afficher le contenu du vecteur
    println!("{:?}", matrix);
}//1 et -1 sur photo -> tous les labels (Y_train)
//X_train 

//0 1 2 3 4 5
//6 7 8 9 0 0