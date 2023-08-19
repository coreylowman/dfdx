#![feature(generic_const_exprs)]

mod batch_norm2d;
mod bias1d;
mod bias2d;
mod conv2d;
mod flatten2d;
mod generalized_add;
mod layer_norm1d;
mod linear;
mod matmul;
mod multi_head_attention;
mod pool_2d_avg;
mod pool_2d_max;
mod pool_2d_min;
mod pool_global_avg;
mod pool_global_max;
mod pool_global_min;
mod relu;
mod reshape;
mod residual_add;
mod sgd;
mod transformer;

pub use dfdx_nn_core::*;
pub use dfdx_nn_derives::*;

pub use batch_norm2d::{BatchNorm2D, BatchNorm2DConfig, BatchNorm2DConstConfig};
pub use bias1d::{Bias1D, Bias1DConfig, Bias1DConstConfig};
pub use bias2d::{Bias2D, Bias2DConfig, Bias2DConstConfig};
pub use conv2d::{Conv2D, Conv2DConfig, Conv2DConstConfig};
pub use flatten2d::Flatten2D;
pub use generalized_add::GeneralizedAdd;
pub use layer_norm1d::{LayerNorm1D, LayerNorm1DConfig, LayerNorm1DConstConfig};
pub use linear::{Linear, LinearConfig, LinearConstConfig};
pub use matmul::{MatMul, MatMulConfig, MatMulConstConfig};
pub use multi_head_attention::{MultiHeadAttention, MultiHeadAttentionConfig};
pub use pool_2d_avg::{AvgPool2D, AvgPool2DConst};
pub use pool_2d_max::{MaxPool2D, MaxPool2DConst};
pub use pool_2d_min::{MinPool2D, MinPool2DConst};
pub use pool_global_avg::AvgPoolGlobal;
pub use pool_global_max::MaxPoolGlobal;
pub use pool_global_min::MinPoolGlobal;
pub use relu::ReLU;
pub use reshape::Reshape;
pub use residual_add::ResidualAdd;
pub use sgd::Sgd;
pub use transformer::{
    DecoderBlock, DecoderBlockConfig, EncoderBlock, EncoderBlockConfig, Transformer,
    TransformerConfig,
};
