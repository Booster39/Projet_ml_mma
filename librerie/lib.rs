
// Methode qui permet de faire la somme de deux entier a et b

#[no_mangle]
pub extern "C" fn my_add(a: i32, b: i32) -> i32 {
    a + b
}


//Cette Methode nous permet de realiser la somme des elements dans un tableau on parcous le tableau  avec des ieterations.

#[no_mangle]
pub extern "C" fn my_sum(arr: *const i32, nb_elems: i32) -> i32 {
    let safe_arr = unsafe {
        std::slice::from_raw_parts(arr, nb_elems as usize)
    };

    safe_arr.iter().sum()
}


//Cette methode nous permet de prendre un entier au paramtre et remplir un vecteur avec les elements de 0  jusqu'a l'entier - 1 en parametre .
#[no_mangle]
pub extern "C" fn count_to_n(n: i32) -> *mut i32 {
    let mut v: Vec<i32> = (0..n).collect();
    let arr_slice = v.as_mut_ptr();

    std::mem::forget(v);

    arr_slice
}

//ça nous permettre de creer un vecteur à partir du pointeur et de la longueur

#[no_mangle]
extern "C" fn delete_int_array(arr: *mut i32, arr_len: i32) {
    unsafe {
        Vec::from_raw_parts(arr, arr_len as usize, arr_len as usize)
    };
}

//Structure qui represente MLP
// w = Point de connexion entre les neurones
// x = valeur entrer du modele
// deltas = represente le gradient pour la rétropropagation

struct MyMLP {
    nb_layer: usize,
    nb_neurons_per_layer: Vec<usize>,
    W: Vec<Vec<Vec<f32>>>,
    X: Vec<Vec<f32>>,
    deltas: Vec<Vec<f32>>,
}


#[no_mangle]
pub extern "C" fn create_mlp_model() -> *mut MyMLP {
    let model = Box::new(MyMLP {
        nb_layer: 0,   // TODO: Initialize
        nb_neurons_per_layer: vec![], // TODO: Initialize
        W: vec![], // TODO: Initialize
        X: vec![], // TODO: Initialize
        deltas: vec![], // TODO: Initialize
    });

    Box::into_raw(model)
}


#[no_mangle]
pub extern "C" fn create_mlp_model(arr: *mut i32, arr_len: i32) -> *mut MyMLP {
    let mut model = Box::new(MyMLP {
        d: Vec::new(),
        L: 0,
        W: Vec::new(),
        X: Vec::new(),
        deltas: Vec::new(),
    });

    model.d = vec![arr_len as usize];
    model.L = arr_len as usize - 1;

    // Initialize W with zeros
    for l in 0..=model.L {
        model.W.push(Vec::new());
        for i in 0..=model.L {
            model.W[l].push(vec![0.0; model.L + 1]);
        }
    }

    // Initialize X with ones
    for l in 0..=model.L {
        model.X.push(Vec::new());
        for j in 0..=model.L {
            model.X[l].push(1.0);
        }
    }

    // Initialize deltas with zeros
    for l in 0..=model.L {
        model.deltas.push(Vec::new());
        for _j in 0..=model.L {
            model.deltas[l].push(0.0);
        }
    }

    let leaked = Box::into_raw(model);
    leaked
}


#[no_mangle]
extern "C" fn train_mlp_model(
    model: *mut MyMLP,
    dataset_inputs: *const f32,
    lines: i32,
    columns: i32,
    dataset_outputs: *const f32,
    output_columns: i32,
    alpha: f32,
    nb_iter: i32,
    is_classification: bool,
) {
    let model_ref = unsafe { &mut *model };

    let inputs: &[Vec<f32>] = unsafe {
        std::slice::from_raw_parts(dataset_inputs, lines as usize)
    };

    let outputs: &[Vec<f32>] = unsafe {
        std::slice::from_raw_parts(dataset_outputs, lines as usize)
    };

    for _ in 0..nb_iter {
        let k = rand::random::<usize>() % lines as usize;
        let input_k = &inputs[k];
        let output_k = &outputs[k];

        model_ref.train(
            input_k,
            output_k,
            is_classification,
            alpha,
            MyMLP::sigmoid,
        );
    }
}

#[no_mangle]
extern "C" fn delete_mlp_model(model: *mut MyMLP) {
    unsafe {
        Box::from_raw(model);
    }
}

#[no_mangle]
extern "C" fn delete_float_array(arr: *mut f32, arr_len: i32) {
    unsafe {
        Vec::from_raw_parts(arr, arr_len as usize, arr_len as usize)
    };
}