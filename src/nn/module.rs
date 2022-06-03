use std::{
    fs::File,
    io::{BufWriter, Seek, Write},
    path::Path,
};
use zip::{result::ZipResult, ZipWriter};

use crate::prelude::CanUpdateWithGradients;

/// A unit of a neural network. Acts on the generic `Input`
/// and produces `Module::Output`.
///
/// Generic `Input` means you can implement module for multiple
/// input types on the same struct. For example [Linear] implements
/// [Module] for 1d inputs and 2d inputs.
pub trait Module<Input>: Default + CanUpdateWithGradients {
    type Output;
    fn forward(&self, input: Input) -> Self::Output;
}

/// Something that can be saved to a `.zip`
///
/// All [Module]s in nn implement SaveToZip by default, and the zips are formatted in a `.npz` fashion.
pub trait SaveToZip {
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
    fn write<W: Write + Seek>(
        &self,
        filename_prefix: &String,
        w: &mut ZipWriter<W>,
    ) -> ZipResult<()>;

    /// Save this object into the `.npz` file determined located at `path`.
    ///
    /// Example:
    /// ```ignore
    /// # use dfdx::prelude::*;
    /// let model: (Linear<5, 10>, Linear<10, 5>) = Default::default();
    /// model.save("tst.npz")?;
    /// ```
    fn save<P: AsRef<Path>>(&self, path: P) -> ZipResult<()> {
        let mut zip = zip::ZipWriter::new(BufWriter::new(File::create(path)?));
        self.write(&"".into(), &mut zip)?;
        zip.finish()?;
        Ok(())
    }
}
