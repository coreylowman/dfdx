use crate::numpy::{self, NpyError, NumpyDtype, NumpyShape, ReadNumbers, WriteNumbers};
use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Seek, Write},
    path::Path,
};
use zip::{
    result::{ZipError, ZipResult},
    ZipArchive, ZipWriter,
};

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
        let f = File::create(path)?;
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
        let f = File::open(path).map_err(|e| NpzError::Npy(NpyError::IoError(e)))?;
        let f = BufReader::new(f);
        let mut zip = ZipArchive::new(f).map_err(NpzError::Zip)?;
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

/// Error that can happen while loading data from a `.npz` zip archive.
pub enum NpzError {
    /// Something went wrong with reading from the `.zip` archive.
    Zip(ZipError),

    /// Something went wrong with loading data from a `.npy` file
    Npy(NpyError),
}

/// Writes `data` to a new file in a zip archive named `filename`.
///
/// Example:
/// ```ignore
/// let mut zip = ZipWriter::new(...);
/// let linear: Linear<5, 2> = Default::default();
/// npz_fwrite(&mut zip, "weight.npy".into(), linear.data());
/// ```
pub fn npz_fwrite<W: Write + Seek, T: NumpyDtype + NumpyShape + WriteNumbers>(
    w: &mut zip::ZipWriter<W>,
    filename: String,
    data: &T,
) -> ZipResult<()> {
    w.start_file(filename, Default::default())?;
    numpy::write(w, data)?;
    Ok(())
}

/// Reads `data` from a file already in a zip archive named `filename`.
///
/// Example:
/// ```ignore
/// let mut zip = ZipArchive::new(...);
/// let mut linear: Linear<5, 2> = Default::default();
/// npz_fread(&mut zip, "weight.npy".into(), linear.weight.mut_data());
/// ```
pub fn npz_fread<R: Read + Seek, T: NumpyDtype + NumpyShape + ReadNumbers>(
    r: &mut zip::ZipArchive<R>,
    filename: String,
    data: &mut T,
) -> Result<(), NpzError> {
    let mut f = r.by_name(&filename).map_err(NpzError::Zip)?;
    numpy::read(&mut f, data).map_err(NpzError::Npy)?;
    Ok(())
}
