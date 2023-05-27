use image::{DynamicImage, GenericImageView};
use ndarray::{Array, Array2, Axis};
use std::path::Path;

fn load_and_preprocess_image(image_path: &str) -> Vec<f32> {
    let img = image::open(&Path::new(image_path)).unwrap();

    let mut img_data: Vec<u8> = img.raw_pixels();
    let img_ndarray = Array::from_shape_vec((img.height() as usize, img.width() as usize, img.color().channel_count() as usize), img_data).unwrap();

    let resized_img_ndarray = imageops::resize(&img_ndarray, 100, 100, imageops::FilterType::Triangle);

    let normalized_img_ndarray = (resized_img_ndarray.mapv(|x| x as f32 / 255.0) - 0.5) * 2.0;

    let mut normalized_img_data: Vec<f32> = Vec::new();
    normalized_img_ndarray.iter().for_each(|&x| normalized_img_data.push(x));

    normalized_img_data
}

fn put_image_in_array() {
    let mut matrix: Vec<Vec<f32>> = Vec::new();
    for conor in 0..30 {
        let mut link: String = "./dataset/conor_mcgregor/";
        link.push_str(&(conor + 1).to_string());
        matrix.push(load_and_preprocess_image(&link));
    }

    let mut vector: Vec<Vec<f32>> = Vec::new();
    for tyson in 0..28 {
        let mut link: String = "./dataset/mike_tyson/";
        link.push_str(&(tyson + 1).to_string());
        vector.push(load_and_preprocess_image(&link));
    }

    let mut final_matrix: Vec<Vec<f32>> = matrix;
    final_matrix.extend(vector);

    let final_matrix = Array2::from_shape_vec((final_matrix.len(), final_matrix[0].len()), final_matrix.into_iter().flatten().collect::<Vec<f32>>()).unwrap();
    let final_matrix = final_matrix.reversed_axes();

    println!("{:?}", final_matrix);
}

fn main() {
    put_image_in_array();
}
//1 et -1 sur photo -> tous les labels (Y_train)
//X_train 

//0 1 2 3 4 5
//6 7 8 9 0 0