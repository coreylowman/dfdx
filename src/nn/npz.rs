use crate::tensor::numpy::NpzError;
use std::{
    io::{BufReader, BufWriter, Read, Seek, Write},
    path::Path,
};
use zip::{result::ZipResult, ZipArchive, ZipWriter};

/// Something that can be saved to a `.npz` (which is a `.zip`).
///
/// All [super::Module]s in nn implement SaveToNpz, and the zips are formatted in a `.npz` fashion.
pub trait SaveToNpz {
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
    fn write<W>(&self, _filename_prefix: &str, _w: &mut ZipWriter<W>) -> ZipResult<()>
    where
        W: Write + Seek,
    {
        Ok(())
    }
}

/// Something that can be loaded from a `.npz` file (which is a `zip` file).
///
/// All [super::Module]s in nn implement LoadFromNpz, and the zips are formatted in a `.npz` fashion.
pub trait LoadFromNpz {
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
    fn read<R>(&mut self, _filename_prefix: &str, _r: &mut ZipArchive<R>) -> Result<(), NpzError>
    where
        R: Read + Seek,
    {
        Ok(())
    }
}
