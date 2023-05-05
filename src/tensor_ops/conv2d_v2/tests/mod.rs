use super::*;
use crate::{tensor_ops::*, tests::*};

macro_rules! data {
    ($Label:tt, $Name:tt) => {
        std::format!(
            "src/tensor_ops/conv2d_v2/tests/data/{}_{}.npy",
            $Label,
            $Name
        )
    };
}

macro_rules! test_case {
    ($Label:tt, $X:ty, $W:ty, $Y:ty, $S:tt, $P:tt, $D:tt, $G:tt) => {
        let dev: TestDevice = Default::default();
        let mut x: Tensor<$X, f32, _> = dev.zeros();
        let mut w: Tensor<$W, f32, _> = dev.zeros();
        let mut y_true: Tensor<$Y, f32, _> = dev.zeros();
        let mut dx_true: Tensor<$X, f32, _> = dev.zeros();
        let mut dw_true: Tensor<$W, f32, _> = dev.zeros();

        x.load_from_npy(data!($Label, "x")).unwrap();
        w.load_from_npy(data!($Label, "w")).unwrap();
        y_true.load_from_npy(data!($Label, "y")).unwrap();
        dx_true.load_from_npy(data!($Label, "x_grad")).unwrap();
        dw_true.load_from_npy(data!($Label, "w_grad")).unwrap();

        let y =
            (x.leaky_trace(), w.clone()).conv2d(Const::<$S>, Const::<$P>, Const::<$D>, Const::<$G>);
        assert_close_to_tensor!(y, y_true, 1e-5);
        let grads = y.square().mean().backward();
        assert_close_to_tensor!(grads.get(&x), dx_true, 1e-5);
        // assert_close_to_tensor!(grads.get(&w), dw_true, 1e-5);
    };
}

#[test]
fn test_conv2d_default() {
    test_case!("default", Rank3<3, 15, 15>, Rank4<5, 3, 4, 4>, Rank3<5, 12, 12>, 1, 0, 1, 1);
}

#[test]
fn test_conv2d_s2() {
    test_case!("s2", Rank3<3, 15, 15>, Rank4<5, 3, 4, 4>, Rank3<5, 6, 6>, 2, 0, 1, 1);
}

#[test]
fn test_conv2d_p1() {
    test_case!("p1", Rank3<3, 15, 15>, Rank4<5, 3, 4, 4>, Rank3<5, 14, 14>, 1, 1, 1, 1);
}

#[test]
fn test_conv2d_d2() {
    test_case!("d2", Rank3<3, 15, 15>, Rank4<5, 3, 4, 4>, Rank3<5, 9, 9>, 1, 0, 2, 1);
}

#[test]
fn test_conv2d_g2() {
    test_case!("g2", Rank3<6, 15, 15>, Rank4<6, 3, 4, 4>, Rank3<6, 12, 12>, 1, 0, 1, 2);
}

#[test]
fn test_conv2d_g3() {
    test_case!("g3", Rank3<18, 15, 15>, Rank4<9, 6, 4, 4>, Rank3<9, 12, 12>, 1, 0, 1, 3);
}

#[test]
fn test_conv2d_all() {
    test_case!("all", Rank3<9, 15, 15>, Rank4<6, 3, 4, 4>, Rank3<6, 6, 6>, 2, 1, 2, 3);
}

// #[test]
// fn test_conv2d_large() {
//     test_case!("large", Rank3<3, 224, 224>, Rank4<64, 3, 7, 7>, Rank3<64, 112, 112>, 2, 3, 1, 1);
// }
