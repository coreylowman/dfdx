use dfdx::numpy as np;

fn main() {
    np::save("0d-rs.npy", &1.234).expect("Saving failed");
    np::save("1d-rs.npy", &[1.0, 2.0, 3.0]).expect("Saving failed");
    np::save("2d-rs.npy", &[[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]).expect("Saving failed");

    let mut expected_0d = 0.0;
    np::load("0d-rs.npy", &mut expected_0d).expect("Loading failed");
    assert_eq!(expected_0d, 1.234);

    let mut expected_1d = [0.0; 3];
    np::load("1d-rs.npy", &mut expected_1d).expect("Loading failed");
    assert_eq!(expected_1d, [1.0, 2.0, 3.0]);

    let mut expected_2d = [[0.0; 3]; 2];
    np::load("2d-rs.npy", &mut expected_2d).expect("Loading failed");
    assert_eq!(expected_2d, [[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
}
