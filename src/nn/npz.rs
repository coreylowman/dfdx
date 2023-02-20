use crate::{
    prelude::{numpy::NumpyDtype, CopySlice, Tensor},
    shapes::{Dtype, Shape},
    tensor::numpy::NpzError,
};
use std::{
    io::{BufReader, BufWriter, Read, Seek, Write},
    path::Path,
    string::ToString,
};
use zip::{
    result::{ZipError, ZipResult},
    ZipArchive, ZipWriter,
};

use super::{DeviceStorage, TensorFunction, TensorFunctionOption, VisitTensors, VisitTensorsMut};

struct SaveToNpzVisitor<'a, W: Write + Seek> {
    writer: &'a mut ZipWriter<W>,
}

impl<W: Write + Seek, E: Dtype + NumpyDtype, D: DeviceStorage + CopySlice<E>>
    TensorFunction<1, 0, E, D> for SaveToNpzVisitor<'_, W>
{
    type Err = ZipError;

    fn call<S: Shape>(
        &mut self,
        refs: [&Tensor<S, E, D>; 1],
        _refs_mut: [&mut Tensor<S, E, D>; 0],
        name: Option<std::string::String>,
        _options: &[TensorFunctionOption],
    ) -> Result<(), Self::Err> {
        refs[0].write_to_npz(self.writer, std::format!("{}.npz", name.unwrap()))
    }
}

/// Something that can be saved to a `.npz` (which is a `.zip`).
///
/// All [super::Module]s in nn implement SaveToNpz, and the zips are formatted in a `.npz` fashion.
pub trait SaveToNpz<E: Dtype + NumpyDtype, D: DeviceStorage + CopySlice<E>>:
    VisitTensors<E, D>
{
    /// Save this object into the `.npz` file determined located at `path`.
    ///
    /// Example:
    /// ```ignore
    /// # use dfdx::prelude::*;
    /// let model: (Linear<5, 10>, Linear<10, 5>) = Default::default();
    /// model.save("tst.npz")?;
    /// ```
    fn save<P: AsRef<Path>>(&self, path: P) -> ZipResult<()> {
        let f = std::fs::File::create(path)?;
        let f = BufWriter::new(f);
        let mut zip = ZipWriter::new(f);
        self.write("", &mut zip)?;
        zip.finish()?;
        Ok(())
    }

    /// Write this object into [ZipWriter] `w` with a base filename of `filename_prefix`.
    ///
    /// Example:
    /// ```ignore
    /// # use dfdx::prelude::*;
    /// let model: Linear<5, 10> = Default::default();
    /// let mut zip = ZipWriter::new(...);
    /// model.write("0.", &mut zip)?;
    /// model.write("1.", &mut zip)?;
    /// ```
    /// Will save a zip file with the following files in it:
    /// - `0.weight.npy`
    /// - `0.bias.npy`
    /// - `1.weight.npy`
    /// - `1.bias.npy`
    fn write<W>(&self, filename_prefix: &str, w: &mut ZipWriter<W>) -> ZipResult<()>
    where
        W: Write + Seek,
    {
        let mut visitor = SaveToNpzVisitor { writer: w };
        self.visit_with_name(filename_prefix.to_string(), &mut visitor)
    }
}

impl<E: Dtype + NumpyDtype, D: DeviceStorage + CopySlice<E>, T: VisitTensors<E, D>> SaveToNpz<E, D>
    for T
{
}

struct LoadFromNpzVisitor<'a, R: Read + Seek> {
    reader: &'a mut ZipArchive<R>,
}

impl<R: Read + Seek, E: Dtype + NumpyDtype, D: DeviceStorage + CopySlice<E>>
    TensorFunction<0, 1, E, D> for LoadFromNpzVisitor<'_, R>
{
    type Err = NpzError;

    fn call<S: Shape>(
        &mut self,
        _refs: [&Tensor<S, E, D>; 0],
        refs_mut: [&mut Tensor<S, E, D>; 1],
        name: Option<std::string::String>,
        _options: &[TensorFunctionOption],
    ) -> Result<(), Self::Err> {
        refs_mut[0].read_from_npz(self.reader, std::format!("{}.npz", name.unwrap()))
    }
}

/// Something that can be loaded from a `.npz` file (which is a `zip` file).
///
/// All [super::Module]s in nn implement LoadFromNpz, and the zips are formatted in a `.npz` fashion.
pub trait LoadFromNpz<E: Dtype + NumpyDtype, D: DeviceStorage + CopySlice<E>>:
    VisitTensorsMut<E, D>
{
    /// Loads data from a `.npz` zip archive at the specified `path`.
    ///
    /// Example:
    /// ```ignore
    /// # use dfdx::prelude::*;
    /// let mut model: (Linear<5, 10>, Linear<10, 5>) = Default::default();
    /// model.load("tst.npz")?;
    /// ```
    fn load<P: AsRef<Path>>(&mut self, path: P) -> Result<(), NpzError> {
        let f = std::fs::File::open(path)?;
        let f = BufReader::new(f);
        let mut zip = ZipArchive::new(f)?;
        self.read("", &mut zip)?;
        Ok(())
    }

    /// Reads this object from a [ZipArchive]. `r` with a base filename of `filename_prefix`.
    ///
    /// Example:
    /// ```ignore
    /// # use dfdx::prelude::*;
    /// let mut model: Linear<5, 10> = Default::default();
    /// let mut zip = ZipArchive::new(...);
    /// model.read("0.", &mut zip)?;
    /// ```
    /// Will try to read data from the following files:
    /// - `0.weight.npy`
    /// - `0.bias.npy`
    fn read<R>(&mut self, filename_prefix: &str, r: &mut ZipArchive<R>) -> Result<(), NpzError>
    where
        R: Read + Seek,
    {
        let mut visitor = LoadFromNpzVisitor { reader: r };
        self.visit_mut_with_name(filename_prefix.to_string(), &mut visitor)
    }
}
