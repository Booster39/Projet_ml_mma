use rand::seq::SliceRandom;
use std::collections::HashSet;

fn data_separation(data: Vec<(Vec<f32>, i32)>) -> (Vec<(Vec<f32>, i32)>, Vec<(Vec<f32>, i32)>) {
    let mut train_data = Vec::new();
    let mut test_data = Vec::new();

    let mut label_data = HashSet::new();

    for (_, label) in &data {
        label_data.insert(label);
    }

    for label in label_data.iter() {
        let mut data_label: Vec<(Vec<f32>, i32)> = data.iter().filter(|(_, l)| *l == *label).cloned().collect();
        data_label.shuffle(&mut rand::thread_rng());

        let split_index = (data_label.len() as f32 * 0.8) as usize;
        let (train_label, test_label) = data_label.split_at(split_index);

        train_data.extend(train_label.to_owned());
        test_data.extend(test_label.to_owned());
    }

    (train_data, test_data)
}