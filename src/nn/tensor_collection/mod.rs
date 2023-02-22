mod collection;
mod visitor;

pub use collection::{TensorCollection, TensorOptions, TensorVisitor};
pub use visitor::{RecursiveWalker, TensorContainer, TensorMut, TensorRef, VisitTensors};
