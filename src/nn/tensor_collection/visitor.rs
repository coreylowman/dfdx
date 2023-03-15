use crate::{
    prelude::Device,
    shapes::{Dtype, Shape},
    tensor::Tensor,
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
pub trait TensorVisitor<E: Dtype, D: Device<E>> {
    /// The type of tensor this struct uses. E.g. [ViewTensorMut], or [ViewTensorRef]
    type Viewer: TensorViewer;
    type Err;
    type E2: Dtype;
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

/// A list of a Module's fields,
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
