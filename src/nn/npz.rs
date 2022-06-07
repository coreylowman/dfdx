use super::*;
use crate::{
    numpy::{self, NpyError},
    prelude::HasArrayData,
};
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
/// All [Module]s in nn implement SaveToNpz, and the zips are formatted in a `.npz` fashion.
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
        self.write(&"".into(), &mut zip)?;
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
    fn write<W>(&self, filename_prefix: &String, w: &mut ZipWriter<W>) -> ZipResult<()>
    where
        W: Write + Seek;
}

/// Something that can be loaded from a `.npz` file (which is a `zip` file).
///
/// All [Module]s in nn implement LoadFromNpz, and the zips are formatted in a `.npz` fashion.
pub trait LoadFromNpz {
    /// Loads data from a `.npz` zip archive at the specified `path`.
    ///
    /// Example:
    /// ```ignore
    /// # use dfdx::prelude::*;
    /// let mut model: (Linear<5, 10>, Linear<10, 5>) = Default::default();
    /// model.load("tst.npz")?;
    /// ``
    fn load<P: AsRef<Path>>(&mut self, path: P) -> Result<(), NpzError> {
        let f = File::open(path).map_err(|e| NpzError::Npy(NpyError::IoError(e)))?;
        let f = BufReader::new(f);
        let mut zip = ZipArchive::new(f).map_err(NpzError::Zip)?;
        self.read(&"".into(), &mut zip)?;
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
    fn read<R>(&mut self, filename_prefix: &String, r: &mut ZipArchive<R>) -> Result<(), NpzError>
    where
        R: Read + Seek;
}

/// Error that can happen while loading data from a `.npz` zip archive.
pub enum NpzError {
    /// Something went wrong with reading from the `.zip` archive.
    Zip(ZipError),

    /// Something went wrong with loading data from a `.npy` file
    Npy(NpyError),
}

macro_rules! empty_npz_impl {
    ($struct_name:ident) => {
        impl SaveToNpz for $struct_name {
            /// Does nothing.
            fn write<W>(&self, _: &String, _: &mut ZipWriter<W>) -> ZipResult<()>
            where
                W: Write + Seek,
            {
                Ok(())
            }
        }

        impl LoadFromNpz for $struct_name {
            /// Does nothing.
            fn read<R>(&mut self, _: &String, _: &mut ZipArchive<R>) -> Result<(), NpzError>
            where
                R: Read + Seek,
            {
                Ok(())
            }
        }
    };
}

// Activations
empty_npz_impl!(ReLU);
empty_npz_impl!(Sin);
empty_npz_impl!(Cos);
empty_npz_impl!(Ln);
empty_npz_impl!(Exp);
empty_npz_impl!(Sigmoid);
empty_npz_impl!(Tanh);
empty_npz_impl!(Square);
empty_npz_impl!(Sqrt);
empty_npz_impl!(Abs);

// Other misc layers.
empty_npz_impl!(Dropout);

impl<const N: usize> SaveToNpz for DropoutOneIn<N> {
    /// Does nothing.
    fn write<W>(&self, _: &String, _: &mut ZipWriter<W>) -> ZipResult<()>
    where
        W: Write + Seek,
    {
        Ok(())
    }
}

impl<const N: usize> LoadFromNpz for DropoutOneIn<N> {
    /// Does nothing.
    fn read<R>(&mut self, _: &String, _: &mut ZipArchive<R>) -> Result<(), NpzError>
    where
        R: Read + Seek,
    {
        Ok(())
    }
}

impl<const I: usize, const O: usize> SaveToNpz for Linear<I, O> {
    /// Saves `self.weight` to `{filename_prefix}weight.npy` and `self.bias` to `{filename_prefix}bias.npy`
    /// using [numpy::write()].
    fn write<W>(&self, filename_prefix: &String, w: &mut zip::ZipWriter<W>) -> ZipResult<()>
    where
        W: Write + Seek,
    {
        w.start_file(format!("{filename_prefix}weight.npy"), Default::default())?;
        numpy::write(w, self.weight.data())?;
        w.start_file(format!("{filename_prefix}bias.npy"), Default::default())?;
        numpy::write(w, self.bias.data())?;
        Ok(())
    }
}

impl<const I: usize, const O: usize> LoadFromNpz for Linear<I, O> {
    /// Reads `self.weight` from `{filename_prefix}weight.npy` and `self.bias` from `{filename_prefix}bias.npy`
    /// using [numpy::read()].
    fn read<R>(&mut self, filename_prefix: &String, r: &mut ZipArchive<R>) -> Result<(), NpzError>
    where
        R: Read + Seek,
    {
        {
            let mut f = r
                .by_name(&format!("{filename_prefix}weight.npy"))
                .map_err(NpzError::Zip)?;
            numpy::read(&mut f, self.weight.mut_data()).map_err(NpzError::Npy)?;
        }

        {
            let mut f = r
                .by_name(&format!("{filename_prefix}bias.npy"))
                .map_err(NpzError::Zip)?;
            numpy::read(&mut f, self.bias.mut_data()).map_err(NpzError::Npy)?;
        }

        Ok(())
    }
}

macro_rules! tuple_impls {
    ([$($name:ident),+] [$($idx:tt),+]) => {
        impl<$($name: SaveToNpz),+> SaveToNpz for ($($name,)+) {
            /// Calls `SaveToNpz::write(self.<idx>, ...)` on each part of the tuple. See [SaveToNpz].
            ///
            /// E.g. for a two tuple (A, B) with `base == ""`, this will call:
            /// 1. `self.0.write("0.", w)`
            /// 2. `self.1.write("1.", w)`
            fn write<W: Write + Seek>(&self, base: &String, w: &mut ZipWriter<W>) -> ZipResult<()> {
                $(self.$idx.write(&format!("{}{}.", base, $idx), w)?;)+
                Ok(())
            }
        }

        impl<$($name: LoadFromNpz),+> LoadFromNpz for ($($name,)+) {
            /// Calls `LoadFromNpz::read(self.<idx>, ...)` on each part of the tuple. See [LoadFromNpz].
            ///
            /// E.g. for a two tuple (A, B) with `base == ""`, this will call:
            /// 1. `self.0.read("0.", r)`
            /// 2. `self.1.read("1.", r)`
            fn read<R: Read + Seek>(&mut self, base: &String, r: &mut ZipArchive<R>) -> Result<(), NpzError> {
                $(self.$idx.read(&format!("{}{}.", base, $idx), r)?;)+
                Ok(())
            }
        }
    };
}

tuple_impls!([A, B] [0, 1]);
tuple_impls!([A, B, C] [0, 1, 2]);
tuple_impls!([A, B, C, D] [0, 1, 2, 3]);
tuple_impls!([A, B, C, D, E] [0, 1, 2, 3, 4]);
tuple_impls!([A, B, C, D, E, F] [0, 1, 2, 3, 4, 5]);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::Randomize;
    use rand::{prelude::StdRng, SeedableRng};
    use rand_distr::Standard;
    use tempfile::NamedTempFile;

    #[test]
    fn test_save_linear() {
        let model: Linear<5, 3> = Default::default();
        let file = NamedTempFile::new().expect("failed to create tempfile");
        model
            .save(file.path().to_str().unwrap().to_string())
            .expect("failed to save model");
        let f = File::open(file.path()).expect("failed to open resulting file");
        let mut zip = ZipArchive::new(f).expect("failed to create zip archive from file");
        {
            let weight_file = zip
                .by_name("weight.npy")
                .expect("failed to find weight.npy file");
            assert!(weight_file.size() > 0);
        }
        {
            let bias_file = zip
                .by_name("bias.npy")
                .expect("failed to find bias.npy file");
            assert!(bias_file.size() > 0);
        }
    }

    #[test]
    fn test_save_tuple() {
        let model: (
            Linear<1, 2>,
            ReLU,
            Linear<2, 3>,
            (Dropout, Linear<1, 2>, Linear<3, 4>),
        ) = Default::default();
        let file = NamedTempFile::new().expect("failed to create tempfile");
        model
            .save(file.path().to_str().unwrap().to_string())
            .expect("failed to save model");
        let f = File::open(file.path()).expect("failed to open resulting file");
        let zip = ZipArchive::new(f).expect("failed to create zip archive from file");
        let mut names = zip.file_names().collect::<Vec<&str>>();
        names.sort();
        assert_eq!(
            &names,
            &[
                "0.bias.npy",
                "0.weight.npy",
                "2.bias.npy",
                "2.weight.npy",
                "3.1.bias.npy",
                "3.1.weight.npy",
                "3.2.bias.npy",
                "3.2.weight.npy",
            ]
        );
    }

    #[test]
    fn test_load_linear() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut saved_model: Linear<5, 3> = Default::default();
        saved_model.randomize(&mut rng, &Standard);

        let file = NamedTempFile::new().expect("failed to create tempfile");
        assert!(saved_model
            .save(file.path().to_str().unwrap().to_string())
            .is_ok());

        let mut loaded_model: Linear<5, 3> = Default::default();
        assert!(loaded_model.weight.data() != saved_model.weight.data());
        assert!(loaded_model.bias.data() != saved_model.bias.data());

        assert!(loaded_model
            .load(file.path().to_str().unwrap().to_string())
            .is_ok());
        assert_eq!(loaded_model.weight.data(), saved_model.weight.data());
        assert_eq!(loaded_model.bias.data(), saved_model.bias.data());
    }

    #[test]
    fn test_load_tuple() {
        type Model = (
            Linear<1, 2>,
            ReLU,
            Linear<2, 3>,
            (Dropout, Linear<1, 2>, Linear<3, 4>),
        );

        let mut rng = StdRng::seed_from_u64(0);
        let mut saved_model: Model = Default::default();
        saved_model.randomize(&mut rng, &Standard);

        let file = NamedTempFile::new().expect("failed to create tempfile");
        assert!(saved_model
            .save(file.path().to_str().unwrap().to_string())
            .is_ok());

        let mut loaded_model: Model = Default::default();
        assert!(loaded_model
            .load(file.path().to_str().unwrap().to_string())
            .is_ok());
        assert_eq!(loaded_model.0.weight.data(), saved_model.0.weight.data());
        assert_eq!(loaded_model.0.bias.data(), saved_model.0.bias.data());
        assert_eq!(loaded_model.2.weight.data(), saved_model.2.weight.data());
        assert_eq!(loaded_model.2.bias.data(), saved_model.2.bias.data());
        assert_eq!(
            loaded_model.3 .1.weight.data(),
            saved_model.3 .1.weight.data()
        );
        assert_eq!(loaded_model.3 .1.bias.data(), saved_model.3 .1.bias.data());

        assert_eq!(
            loaded_model.3 .2.weight.data(),
            saved_model.3 .2.weight.data()
        );
        assert_eq!(loaded_model.3 .2.bias.data(), saved_model.3 .2.bias.data());
    }
}
