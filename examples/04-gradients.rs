//! Intro to Gradients and Tapes

use dfdx::{
    nn::ZeroGrads,
    shapes::{Rank0, Rank2},
    tensor::{AsArray, AutoDevice, Gradients, NoneTape, OwnedTape, SampleTensor, Tensor, Trace},
    tensor_ops::{Backward, MeanTo, TryMatMul},
};

fn main() {
    let dev = AutoDevice::default();

    // tensors are actually generic over a fourth field: the tape
    // by default tensors are created with a `NoneTape`, which
    // means they don't currently **own** a tape.
    let weight: Tensor<Rank2<4, 2>, f32, _, NoneTape> = dev.sample_normal();
    let a: Tensor<Rank2<3, 4>, f32, _, NoneTape> = dev.sample_normal();

    // the first step to tracing is to allocate some gradients
    // for the tensors we want to track
    let grads = a.alloc_grads();

    // then we pass these gradients into the tensor with .trace().
    // NOTE: the tape has changed from a `NoneTape` to an `OwnedTape`.
    let b: Tensor<Rank2<3, 4>, _, _, OwnedTape<f32, AutoDevice>> = a.trace(grads);

    // the tape will **automatically** be moved around as you perform ops
    // ie. the tapes on inputs to operations are moved to the output
    // of the operation.
    let c: Tensor<Rank2<3, 2>, _, _, OwnedTape<f32, AutoDevice>> = b.matmul(weight.clone());
    let d: Tensor<Rank2<3, 2>, _, _, OwnedTape<f32, AutoDevice>> = c.sin();
    let e: Tensor<Rank0, _, _, OwnedTape<f32, AutoDevice>> = d.mean();

    // finally you can use .backward() to extract the gradients!
    // NOTE: that this method is only available on tensors that **own**
    //       the tape!
    let gradients: Gradients<f32, AutoDevice> = e.backward();

    // now you can extract gradients for specific tensors
    // by querying with them
    let weight_grad: [[f32; 2]; 4] = gradients.get(&weight).array();
    dbg!(weight_grad);

    let a_grad: [[f32; 4]; 3] = gradients.get(&a).array();
    dbg!(a_grad);
}
