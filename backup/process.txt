use image::{DynamicImage, GenericImageView, Pixel};
use std::path::Path;
use ndarray::{Array3, Array2, s};



pub fn load_and_preprocess_image(image_path: &str) -> Array3<f32> {
    // Permet d'ouvrire l'image et de la charger en mémoire
    let img = image::open(&Path::new(image_path)).expect("Failed to open image");

    // Charge l'image dynamiquement dans la mémoire
    let dyn_img: DynamicImage = img.into();

    // Resize the image to 32x32 pixels
    let resized_img = dyn_img.resize_exact(32, 32, image::imageops::Triangle);

    // Division de l'image en 1 tableau de 3 dimensions
    let (width, height) = resized_img.dimensions();
    let mut img_ndarray = Array3::<f32>::zeros((height as usize, width as usize, 3));
    resized_img.pixels().for_each(|(x, y, pixel)| {
        let channels = pixel.channels();
        let pixel_value: [f32; 3] = [
            channels[0] as f32 / 255.0,
            channels[1] as f32 / 255.0,
            channels[2] as f32 / 255.0,
        ];
        let pixel_array = ndarray::array![pixel_value].into_shape((3)).unwrap();
        let mut view = img_ndarray.slice_mut(s![y as usize, x as usize, ..]);
        view.assign(&pixel_array);
    });

   /*  println!("Valeurs de img_ndarray:");
    for row in img_ndarray.outer_iter() {
        for pixel in row.iter() {
            print!("{:?} ", pixel);
        }
        println!();
    }
*/
    img_ndarray
}

pub fn put_image_in_array() {
    let num_rows = 2;
    let num_cols = 27;

    // Créer un vecteur de 2 lignes
    let mut matrix: Array2<Vec<Array3<f32>>> = Array2::from_elem((num_rows, num_cols), Vec::new());

    // Remplir chaque ligne avec 17 tableaux Array3<f32>
    for i in 0..num_rows {
        for j in 0..num_cols {
            if (i == 0) {
                let image_path: String = "dataset/conor_mcgregor/".to_owned() + (i+1).to_string().as_str() + ".jpg";
                matrix[[i, j]].push(load_and_preprocess_image(image_path.as_str()));
            } else {
                let image_path: String = "dataset/mike_tyson/".to_owned() + (i+1).to_string().as_str() + ".jpg";
                matrix[[i, j]].push(load_and_preprocess_image(image_path.as_str()));
            }
        }
    }

    // Afficher le contenu du vecteur
    for row in matrix.rows() {
        for arr in row {
            println!("{:?}", arr);
        }
        println!(); // Sauter une ligne entre chaque ligne
    }

}

//1 et -1 sur photo -> tous les labels (Y_train)
//X_train 

//0 1 2 3 4 5
//6 7 8 9 0 0