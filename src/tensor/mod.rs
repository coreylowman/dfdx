mod impl_binary_ops;
mod impl_can_update_with_tape;
mod impl_default;
mod impl_duplicate;
mod impl_has_tape_holder;
mod impl_has_unique_id;
mod impl_randomize;
mod impl_shaped_array;
mod impl_tape_creator;
mod impl_tensor;
mod impl_tensor_creator;
mod impl_unary_ops;
mod structs;
mod tape_holders;

pub use impl_binary_ops::*;
pub use impl_can_update_with_tape::*;
pub use impl_default::*;
pub use impl_duplicate::*;
pub use impl_has_tape_holder::*;
pub use impl_has_unique_id::*;
pub use impl_randomize::*;
pub use impl_shaped_array::*;
pub use impl_tape_creator::*;
pub use impl_tensor::*;
pub use impl_tensor_creator::*;
pub use impl_unary_ops::*;
pub use structs::*;
pub use tape_holders::*;

pub fn backward<T: Tensor<TapeHolder = WithTape>>(t: T) -> Box<crate::gradients::GradientTape> {
    let id = t.id();
    let (_, mut tape_holder) = t.split_tape_holder();
    tape_holder.0.backward(id);
    tape_holder.0
}
