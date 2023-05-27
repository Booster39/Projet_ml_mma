use std::path::Path;
use ndarray::{Array3, ArrayViewMut3, s};
use image::{DynamicImage, GenericImageView, Pixel};


fn load_and_preprocess_image(image_path: &str) -> Array3<f32> {
    // Charger l'image à partir de son fichier
    let img = image::open(&Path::new(image_path)).expect("Failed to open image");

    // Convertir l'image en image dynamique pour la redimensionner
    let dyn_img: DynamicImage = img.into();

    // Redimensionne limage
    let resized_img = dyn_img.resize_exact(100, 100, image::imageops::Triangle);

    // Convertis l'image redimensionné en tableau ndarray
    let (width, height) = resized_img.dimensions();
    let mut img_ndarray = Array3::<f32>::zeros((height as usize, width as usize, 3));
    resized_img.pixels().for_each(|(x, y, pixel)| {
        let channels = pixel.channels();
        let pixel_value: [f32; 3] = [
            channels[0] as f32 / 255.0,
            channels[1] as f32 / 255.0,
            channels[2] as f32 / 255.0,
        ];
        let pixel_array = Array3::from_shape_vec((1, 1, 3), pixel_value.to_vec()).unwrap();
        let mut view = img_ndarray.slice_mut(s![y as usize, x as usize, ..]);
        view.assign(&pixel_array);
    });

    img_ndarray
}

/*fn main() {
    let image_path = "C:/Users/abdoulaye.doucoure/Desktop/Projet_ml_mma/dataset/conor_mcgregor/1.jpg";
    let preprocessed_image = load_and_preprocess_image(image_path);

    // Use the preprocessed image for machine learning
    // ...
}

main()
*/

/*fn put_image_in_array() {
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
*/