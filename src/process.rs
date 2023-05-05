use image::{DynamicImage, GenericImageView};
use ndarray::prelude::*;
use std::path::Path;

fn load_and_preprocess_image(image_path: &str, output_size: (usize, usize)) -> Array3<f32> {
    // Charger une image à partir d'un fichier
    let img = image::open(&Path::new(image_path)).unwrap();

    // Convertir l'image en RVB et la redimensionner
    let resized_img = img.resize_exact(output_size.0 as u32, output_size.1 as u32, image::imageops::FilterType::Nearest);

    // Convertir l'image en tableau et normaliser les pixels
    let img_array = Array3::from_shape_fn((output_size.0, output_size.1, 3), |(i, j, c)| {
        resized_img.get_pixel(i as u32, j as u32)[c] as f32 / 255.0
    });

    // Soustraire la moyenne et diviser par l'écart-type
    let mean: f32 = img_array.iter().sum::<f32>() / img_array.len() as f32;
    let std: f32 = (img_array.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / img_array.len() as f32).sqrt();
    let norm_img_array = (img_array - mean) / std;

    norm_img_array
}