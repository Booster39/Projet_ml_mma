use std::fs::File;                                        // Importation du module File pour la gestion des fichiers
use std::io::prelude::*;                                  // Importation du module io pour les opérations d'entrée/sortie
use std::io::BufReader;                                   // Importation du module BufReader pour la lecture de fichiers ligne par ligne



pub struct MLP {
    d: Vec<i32>,                                          // Vecteur représentant les dimensions des couches du MLP
    l: i32,                                              // Nombre de couches du MLP (longueur du vecteur d)
    w: Vec<Vec<Vec<f64>>>,                                // Matrices de poids du MLP
    x: Vec<Vec<f64>>,                                     // Valeurs des neurones du MLP
    deltas: Vec<Vec<f64>>,                                // Valeurs des deltas pour la rétropropagation
}

#[no_mangle]
pub extern "C" fn createMLP(npl: *const i32, npl_size: i32) -> *mut MLP {
    let mut mlp = Box::new(MLP {                          // Création d'une boîte (box) contenant une instance de MLP
        d: Vec::new(),                                    // Initialisation du vecteur d
        l: npl_size - 1,                                  // Initialisation de L en fonction de la taille de npl
        w: Vec::new(),                                    // Initialisation du vecteur de matrices de poids W
        x: Vec::new(),                                    // Initialisation du vecteur de valeurs de neurones X
        deltas: Vec::new(),                               // Initialisation du vecteur de valeurs de deltas
    });

    unsafe {
        for i in 0..npl_size {                             // Parcours des éléments de npl
            let val = *npl.offset(i as isize);             // Récupération de la valeur à l'index i de npl
            mlp.d.push(val);                               // Ajout de la valeur dans le vecteur d
        }

        for i in 0..mlp.l {                                // Parcours des couches du MLP
            let layer = (0..mlp.d[(i + 1) as usize])        // Génération des matrices de poids pour la couche
                .map(|_| (0..(mlp.d[i as usize] + 1)).map(|_| rand::random::<f64>() * 2.0 - 1.0).collect())
                .collect();
            mlp.w.push(layer);                              // Ajout de la matrice de poids dans le vecteur W
            mlp.deltas.push(vec![0.0; mlp.d[(i + 1) as usize] as usize]);  // Ajout du vecteur de deltas initialisé à 0.0
        }

        for i in 0..(mlp.l + 1) {                          // Parcours des couches du MLP + la couche d'entrée
            mlp.x.push(vec![0.0; (mlp.d[i as usize] + 1) as usize]);  // Ajout du vecteur de neurones initialisé à 0.0
        }
    }

    Box::into_raw(mlp)                                     // Conversion de la boîte en pointeur brut et renvoi du pointeur
}

#[no_mangle]
pub extern "C" fn saveModel(mlp: *const MLP) {
    unsafe {
        if let Some(mlp) = mlp.as_ref() {                   // Vérification que le pointeur MLP n'est pas nul
            let mut file = File::create("model.txt").expect("Erreur lors de la création du fichier model.txt");  // Création du fichier model.txt

            file.write_fmt(format_args!("{}\n", mlp.l)).expect("Erreur lors de l'écriture dans le fichier model.txt");  // Écriture du nombre de couches L dans le fichier

            for i in 0..=(mlp.l as usize) {                  // Parcours des couches du MLP
                file.write_fmt(format_args!("{} ", mlp.d[i])).expect("Erreur lors de l'écriture dans le fichier model.txt");  // Écriture des dimensions des couches d dans le fichier
            }

            file.write_fmt(format_args!("\n")).expect("Erreur lors de l'écriture dans le fichier model.txt");  // Écriture d'une nouvelle ligne dans le fichier

            for i in 0..(mlp.l as usize) {                   // Parcours des couches du MLP
                for j in 0..(mlp.d[(i + 1) as usize] as usize) {  // Parcours des neurones de la couche suivante
                    for k in 0..=(mlp.d[i as usize] as usize) {  // Parcours des neurones de la couche actuelle + biais
                        file.write_fmt(format_args!("{} ", mlp.w[i][j][k])).expect("Erreur lors de l'écriture dans le fichier model.txt");  // Écriture des poids dans le fichier
                    }
                    file.write_fmt(format_args!("\n")).expect("Erreur lors de l'écriture dans le fichier model.txt");  // Écriture d'une nouvelle ligne dans le fichier
                }
            }
        } else {
            eprintln!("Le pointeur MLP est nul");
        }
    }
}

#[no_mangle]
pub extern "C" fn loadModel() -> *mut MLP {
    let mut mlp: *mut MLP = std::ptr::null_mut();

    // Ouverture du fichier "model.txt" en lecture
    let file = File::open("model.txt").expect("Erreur lors de l'ouverture du fichier model.txt");
    let reader = BufReader::new(file);

    // Création d'un itérable des lignes du fichier avec gestion des erreurs
    let mut lines = reader.lines().map(|line| line.expect("Erreur lors de la lecture d'une ligne du fichier"));

    if let Some(line) = lines.next() {
        // Création d'une boîte (Box) pour stocker le MLP chargé à partir du fichier
        let mut mlp_box = Box::new(MLP {
            d: Vec::new(),
            // Conversion de la première ligne en entier pour obtenir la valeur de l
            l: line.parse().expect("Erreur lors de la conversion de la ligne l en entier"),
            w: Vec::new(),
            x: Vec::new(),
            deltas: Vec::new(),
        });

        if let Some(line) = lines.next() {
            // Parcours des valeurs séparées par des espaces sur la deuxième ligne pour remplir le vecteur d
            for val in line.split_whitespace() {
                mlp_box.d.push(val.parse().expect("Erreur lors de la conversion d'une valeur de d en entier"));
            }
        }

        for _ in 0..mlp_box.l {
            let mut layer = Vec::new();
            let mut delta_layer = Vec::new();

            for _ in 0..mlp_box.d[(mlp_box.l as usize) + 1] {
                let mut neuron = Vec::new();

                if let Some(line) = lines.next() {
                    // Parcours des valeurs séparées par des espaces sur les lignes suivantes pour remplir les poids w du MLP
                    for val in line.split_whitespace() {
                        neuron.push(val.parse().expect("Erreur lors de la conversion d'une valeur de w en nombre à virgule flottante"));
                    }
                }

                layer.push(neuron);
                delta_layer.push(0.0);
            }

            mlp_box.w.push(layer);
            mlp_box.deltas.push(delta_layer);
        }

        for _ in 0..=(mlp_box.l as usize) {
            mlp_box.x.push(Vec::new());
        }

        // Conversion de la boîte (Box) en pointeur brut (raw pointer)
        mlp = Box::into_raw(mlp_box);
    }

    mlp
}

#[no_mangle]
pub extern "C" fn propagate(mlp: *mut MLP, inputs: *const f64, is_classification: bool) {
    unsafe {
        for i in 0..(*mlp).d[0] as usize {
            (*mlp).x[0][i] = *inputs.add(i);
        }

        (*mlp).x[0][(*mlp).d[0] as usize] = 1.0;

        for i in 1..=(*mlp).l as usize {
            for j in 0..(*mlp).d[i] as usize {
                let mut sum = 0.0;

                for k in 0..=(*mlp).d[i - 1] as usize {
                    sum += (*mlp).w[i - 1][j][k] * (*mlp).x[i - 1][k];
                }

                (*mlp).x[i][j] = 1.0 / (1.0 + f64::exp(-sum));
            }

            if i != (*mlp).l as usize || !is_classification {
                (*mlp).x[i][(*mlp).d[i] as usize] = 1.0;
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn predict_pmc(mlp: *mut MLP, inputs: *const f64, _inputs_size: i32, is_classification: bool, output: *mut f64, outputs_size: i32) {
    unsafe {
        propagate(mlp, inputs, is_classification);
        // Copie des sorties dans le tableau de sortie
        for i in 0..outputs_size {
            *output.offset(i as isize) = (*mlp).x[(*mlp).l as usize][i as usize];
        }
    }
}


#[no_mangle]
pub extern "C" fn train(mlp: *mut MLP, samples_inputs: *const f64, samples_expected_outputs: *const f64,
                        samples_size: i32, inputs_size: i32, outputs_size: i32,
                        is_classification: bool, iteration_count: i32, alpha: f64) {
    unsafe {
        let mut loss = 0.0;
        let mut accuracy = 0.0;
        let mut best_accuracy = 0.0; // Variable pour conserver la meilleure précision

        for i in 0..iteration_count {
            let index = (rand::random::<usize>() % samples_size as usize) as usize;
            let input_ptr = samples_inputs.add((index * inputs_size as usize) as usize);
            propagate(mlp, input_ptr, is_classification);

            for j in 0..outputs_size as usize {
                let delta = (*samples_expected_outputs.add((index * outputs_size as usize + j) as usize) - (*mlp).x[(*mlp).l as usize][j])
                    * (*mlp).x[(*mlp).l as usize][j] * (1.0 - (*mlp).x[(*mlp).l as usize][j]);
                (*mlp).deltas[(*mlp).l as usize - 1][j] = delta;
                loss += f64::powf(*samples_expected_outputs.add(index * outputs_size as usize + j) - (*mlp).x[(*mlp).l as usize][j], 2.0);

                if is_classification {
                    if f64::round((*mlp).x[(*mlp).l as usize][j]) == *samples_expected_outputs.add(index * outputs_size as usize + j) {
                        accuracy += 1.0;
                    }
                }
            }

            for j in (0..(*mlp).l as usize - 1).rev() {
                for k in 0..(*mlp).d[j + 1] as usize {
                    let mut sum = 0.0;

                    for l in 0..(*mlp).d[j + 2] as usize {
                        sum += (*mlp).w[j + 1][l][k] * (*mlp).deltas[j + 2][l]; // Corrected index here
                    }

                    (*mlp).deltas[j][k] = sum * (*mlp).x[j + 1][k] * (1.0 - (*mlp).x[j + 1][k]);
                }
            }



            for j in 0..(*mlp).l as usize {
                for k in 0..(*mlp).d[j + 1] as usize {
                    for l in 0..(*mlp).d[j] as usize {
                        if l == (*mlp).d[j] as usize - 1 {
                            (*mlp).w[j][k][l] += alpha * (*mlp).deltas[j][k];
                        } else {
                            (*mlp).w[j][k][l] += alpha * (*mlp).deltas[j][k] * (*mlp).x[j][l];
                        }
                    }
                }
            }

            if i % 500 == 0 {
                loss = loss / (outputs_size as f64 * 500.0);
                accuracy = accuracy / (outputs_size as f64 * 500.0);

                // Mise à jour de la meilleure précision atteinte
                if accuracy > best_accuracy {
                    best_accuracy = accuracy;
                }

                // Réinitialiser la perte et l'exactitude pour le prochain lot de 500 itérations
                loss = 0.0;
                accuracy = 0.0;
            }
        }

        println!("Meilleure précision atteinte : {:.2}%", best_accuracy * 100.0);

        saveModel(mlp);
    }
}

#[no_mangle]
pub extern "C" fn deleteMLP(mlp: *mut MLP) {
    unsafe {
        if !mlp.is_null() {
            // Utilisation d'un soulignement pour ignorer la valeur de retour de Box::from_raw(mlp)
            let _ = Box::from_raw(mlp);
        }
    }
}

