import os
import ctypes
import numpy as np

# Charger la bibliothèque Rust en objet partagé (shared object)

lib_path = "/Users/thomas/Documents/Ecole/3A_ESGI/Projet_A/Projet_ml_mma/pmc_final/target/release/libpmc_final.dylib"
lib_path_str = os.fsencode(lib_path)  # Convertir le chemin d'accès en une chaîne d'octets
mlp_dll = ctypes.CDLL(lib_path_str.decode())  # Convertir la chaîne d'octets en une chaîne de caractères et charger la bibliothèque


# Définir les types pour les arguments et les valeurs de retour

mlp_dll.createMLP.restype = ctypes.c_void_p
mlp_dll.createMLP.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
mlp_dll.train.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                          ctypes.c_int, ctypes.c_int, ctypes.c_int,ctypes.c_bool, ctypes.c_int, ctypes.c_double]
mlp_dll.predict_pmc.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_int,
                            ctypes.c_bool, ctypes.POINTER(ctypes.c_double), ctypes.c_int]


test_1_all_samples_inputs = [
    [0, 0],
    [0, 1],
    [1, 0]
]
test_1_all_samples_expected_outputs = [
    [1],
    [-1],
    [-1]
]

# Création du MLP avec 2 entrées, 1 neurone caché et 1 sortie
npl = np.array([2, 1], dtype=np.int32)
mlp_ptr = mlp_dll.createMLP(npl.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), npl.size)


# Assurez-vous que les données d'entrée sont correctement transposées pour correspondre aux attentes de la fonction train en Rust
samples_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.double)
samples_expected_outputs = np.array([[0], [1], [1], [0]], dtype=np.double)

# Appel de la fonction train en utilisant les bonnes dimensions des tableaux
mlp_dll.train(mlp_ptr,
               samples_inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
               samples_expected_outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
               samples_inputs.shape[0], samples_inputs.shape[1], samples_expected_outputs.shape[1],
               True, 400000, 0.02)

input_t = np.zeros(2, dtype=np.double)
output = np.zeros(1, dtype=np.double)
for i in range(samples_inputs.shape[0]):
    input_t[0] = samples_inputs[i, 0]
    input_t[1] = samples_inputs[i, 1]
    mlp_dll.predict_pmc(mlp_ptr,
                    input_t.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    input_t.size,
                    True,
                    output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    output.size)
    print(f"[{int(input_t[0])}, {int(input_t[1])}] = {output[0]}")