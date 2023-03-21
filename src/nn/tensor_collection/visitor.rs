use crate::{
    shapes::{Dtype, Shape},
    tensor::Tensor,
    tensor_ops::Device,
};

use super::{ModuleVisitor, TensorCollection, TensorOptions};

/// A standard [ModuleVisitor] that executes `F` on every [Tensor] encountered.
/// `F` must implement [TensorVisitor]
#[derive(Debug)]
pub struct RecursiveWalker<'a, M, F> {
    pub m: M,
    pub f: &'a mut F,
}

/// Something that can visit [Tensor]s. Used in conjunction with [RecursiveWalker].
///
/// Example implementation to add two Modules together:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// // A TensorVisitor that will add two Modules together, returning the resulting module.
/// struct Adder;
///
/// impl<E: Dtype, D: Device<E>> TensorVisitor<E, D> for Adder {
///     // Take a tuple of references to tensors
///     type Viewer = (ViewTensorRef, ViewTensorRef);
///     type Err = D::Err;
///
///     // Output with the device and dtype that are given
///     type E2 = E;
///     type D2 = D;
///
///     fn visit<S: Shape>(
///         &mut self,
///         opts: TensorOptions<S, E, D>,
///         (a, b): (&Tensor<S, E, D>, &Tensor<S, E, D>),
///     ) -> Result<Option<Tensor<S, Self::E2, Self::D2>>, Self::Err> {
///         // Returns Ok(Some(_)) to construct an output module. Return Ok(None) to not construct
///         // an output
///         Ok(Some(a.clone().try_add(b.clone())?))
///     }
/// }
///
/// type Model = Linear<2, 5>;
/// let model1 = dev.build_module::<Model, f32>();
/// let model2 = dev.build_module::<Model, f32>();
/// let model3 = TensorCollection::iter_tensors(&mut RecursiveWalker {
///     m: (&model1, &model2),
///     f: &mut Adder,
/// }).unwrap().unwrap();
///
/// assert_eq!(
///     (model1.weight.clone() + model2.weight.clone()).array(),
///     model3.weight.array()
/// );
/// assert_eq!(
///     (model1.bias.clone() + model2.bias.clone()).array(),
///     model3.bias.array()
/// );
/// ```
pub trait TensorVisitor<E: Dtype, D: Device<E>> {
    /// The type of tensor this struct uses. E.g. [ViewTensorMut], or [ViewTensorRef]
    type Viewer: TensorViewer;
    type Err;
    /// The dtype to output with
    type E2: Dtype;
    /// The device to output with
    type D2: Device<Self::E2>;

    /// What to do when visiting each Tensor. Return `Ok(None)` if this visitor should not
    /// construct a new module each time it is used, and `Ok(Some(_))` if it should.
    #[allow(clippy::type_complexity)]
    fn visit<S: Shape>(
        &mut self,
        opts: TensorOptions<S, E, D>,
        t: <Self::Viewer as TensorViewer>::View<'_, Tensor<S, E, D>>,
    ) -> Result<Option<Tensor<S, Self::E2, Self::D2>>, Self::Err>;
}

/// Something that can view [Tensor]s in different ways. For example
/// [ViewTensorRef] can view `&Tensor`, and [ViewTensorMut] can view `&mut Tensor.
pub trait TensorViewer: 'static {
    type View<'a, Mod: 'a>
    where
        Self: 'a;

    /// Given a view of a module, returns a view of one of that module's fields
    fn view_field<'a, Mod, Field, GetRef, GetMut>(
        module: &'a mut Self::View<'_, Mod>,
        name: &str,
        get_ref: &mut GetRef,
        get_mut: &mut GetMut,
    ) -> Self::View<'a, Field>
    where
        GetRef: FnMut(&Mod) -> &Field,
        GetMut: FnMut(&mut Mod) -> &mut Field;
}

/// A list of a Module's fields. Used in [ModuleVisitor::visit_fields].
pub trait ModuleFields<M: TensorCollection<E, D>, E: Dtype, D: Device<E>> {
    /// A list of optional instances of each field,
    type Options<E2: Dtype, D2: Device<E2>>;

    /// A list of instances of each field,
    type Output<E2: Dtype, D2: Device<E2>>;

    /// Calls [ModuleVisitor::visit_module] or [ModuleVisitor::visit_tensor] for each field,
    /// and returns optionally constructed fields
    fn visit_fields<V: ModuleVisitor<M, E, D>>(
        self,
        module: &mut V,
    ) -> Result<Self::Options<V::E2, V::D2>, V::Err>;

    /// If any optional fields are None, returns None. Otherwise returns instances of all fields.
    fn handle_options<E2: Dtype, D2: Device<E2>>(
        options: Self::Options<E2, D2>,
    ) -> Option<Self::Output<E2, D2>>;
}

/// A [ModuleFields] that represents a field that contains one or more Tensors.
pub struct ModuleField<'a, F1, F2, Mod, Field>
where
    F1: FnMut(&Mod) -> &Field,
    F2: FnMut(&mut Mod) -> &mut Field,
{
    pub(super) name: &'a str,
    pub(super) get_ref: F1,
    pub(super) get_mut: F2,
    pub(super) m: std::marker::PhantomData<Mod>,
    pub(super) f: std::marker::PhantomData<Field>,
}

/// A [ModuleFields] that represents a field that contains a single Tensor.
pub struct TensorField<'a, F1, F2, Mod, S: Shape, E: Dtype, D: Device<E>>
where
    F1: FnMut(&Mod) -> &Tensor<S, E, D>,
    F2: FnMut(&mut Mod) -> &mut Tensor<S, E, D>,
{
    pub(super) name: &'a str,
    pub(super) get_ref: F1,
    pub(super) get_mut: F2,
    pub(super) options: TensorOptions<S, E, D>,
    pub(super) m: std::marker::PhantomData<Mod>,
}

/// A [TensorViewer] that represents a `&Tensor`
#[derive(Debug)]
pub enum ViewTensorRef {}

/// A [TensorViewer] that represents a `&mut Tensor`
#[derive(Debug)]
pub enum ViewTensorMut {}

/// A [TensorViewer] that represents a Tensor's name as a `String`
#[derive(Debug)]
pub enum ViewTensorName {}
