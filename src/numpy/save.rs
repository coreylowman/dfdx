//! Provides some generic functions to save Nd arrays in the .npy format.

use super::*;
use std::{
    fs::File,
    io::{BufWriter, Result, Write},
    path::Path,
};

/// Saves the data to a .npy file. This calls [write()].
///
/// This is implemented for an arbitrarily shaped array.
/// See [WriteNumbers] for how this is done (recursive array traits!).
///
/// Currently only implemented for f32 and f64 arrays. To add another
/// base type, you can implement [NumpyShape]
///
/// Example Usage:
/// ```ignore
/// use dfdx::numpy;
/// let arr = [[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
/// numpy::save("test.npy", &arr);
/// ```
///
/// Returns the number of bytes written to the file.
pub fn save<T, P>(path: P, t: &T) -> Result<()>
where
    T: NumpyDtype + NumpyShape + WriteNumbers,
    P: AsRef<Path>,
{
    let mut f = BufWriter::new(File::create(path)?);
    write(&mut f, t)
}

/// Writes the data to a [Write].
///
/// This is implemented for an arbitrarily shaped array.
/// See [WriteNumbers] for how this is done (recursive array traits!).
pub fn write<T, W>(w: &mut W, t: &T) -> Result<()>
where
    T: NumpyDtype + NumpyShape + WriteNumbers,
    W: Write,
{
    write_header::<T, W>(w, Endian::Little)?;
    t.write_numbers(w, Endian::Little)?;
    Ok(())
}

fn write_header<T, W>(w: &mut W, endian: Endian) -> Result<()>
where
    T: NumpyDtype + NumpyShape,
    W: Write,
{
    let shape_str = to_shape_str(T::shape());

    let mut header: Vec<u8> = Vec::new();
    write!(
        &mut header,
        "{{'descr': '{}{}', 'fortran_order': False, 'shape': ({}), }}",
        match endian {
            Endian::Big => '>',
            Endian::Little => '<',
            Endian::Native => '=',
        },
        T::DTYPE,
        shape_str,
    )?;

    // padding
    while (header.len() + 1) % 64 != 0 {
        header.write_all(b"\x20")?;
    }

    // new line termination
    header.write_all(b"\n")?;

    // header length
    assert!(header.len() < u16::MAX as usize);
    assert!(header.len() % 64 == 0);

    w.write_all(MAGIC_NUMBER)?; // magic number
    w.write_all(VERSION)?; // version major & minor
    w.write_all(&(header.len() as u16).to_le_bytes())?;
    w.write_all(&header)?;
    Ok(())
}

/// Write all the numbers in &self with the bytes layed out in [Endian] order.
/// Most types that this should be implemented for have `.to_be_bytes()`, `.to_le_bytes()`,
/// and `.to_ne_bytes()`.
pub trait WriteNumbers {
    fn write_numbers<W: Write>(&self, w: &mut W, endian: Endian) -> Result<()>;
}

impl WriteNumbers for f32 {
    fn write_numbers<W: Write>(&self, w: &mut W, endian: Endian) -> Result<()> {
        match endian {
            Endian::Big => w.write_all(&self.to_be_bytes()),
            Endian::Little => w.write_all(&self.to_le_bytes()),
            Endian::Native => w.write_all(&self.to_ne_bytes()),
        }
    }
}

impl WriteNumbers for f64 {
    fn write_numbers<W: Write>(&self, w: &mut W, endian: Endian) -> Result<()> {
        match endian {
            Endian::Big => w.write_all(&self.to_be_bytes()),
            Endian::Little => w.write_all(&self.to_le_bytes()),
            Endian::Native => w.write_all(&self.to_ne_bytes()),
        }
    }
}

impl<T: WriteNumbers, const M: usize> WriteNumbers for [T; M] {
    fn write_numbers<W: Write>(&self, w: &mut W, endian: Endian) -> Result<()> {
        for self_i in self.iter() {
            self_i.write_numbers(w, endian)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use tempfile::NamedTempFile;

    use super::*;
    use std::process::Command;

    fn describe(p: &Path) -> String {
        let output = Command::new("python")
            .arg("-c")
            .arg(format!(
                "import numpy;a=numpy.load({:?});print(a.dtype, a.shape);print(a)",
                p.as_os_str(),
            ))
            .output()
            .expect("Creating sub process failed");
        assert!(output.stderr.is_empty());
        String::from_utf8(output.stdout).expect("")
    }

    #[test]
    fn test_0d_f32_save() {
        let data: f32 = 0.0;

        let file = NamedTempFile::new().expect("failed to create tempfile");

        save(file.path(), &data).expect("Saving failed");
        assert_eq!(
            describe(file.path()).replace("\r\n", "\n"),
            "float32 ()\n0.0\n"
        );
    }

    #[test]
    fn test_1d_f32_save() {
        let data: [f32; 5] = [0.0, 1.0, 2.0, 3.0, -4.0];

        let file = NamedTempFile::new().expect("failed to create tempfile");

        save(file.path(), &data).expect("Saving failed");
        assert_eq!(
            describe(file.path()).replace("\r\n", "\n"),
            "float32 (5,)\n[ 0.  1.  2.  3. -4.]\n"
        );
    }

    #[test]
    fn test_2d_f32_save() {
        let data: [[f32; 3]; 2] = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]];

        let file = NamedTempFile::new().expect("failed to create tempfile");

        save(file.path(), &data).expect("Saving failed");
        assert_eq!(
            describe(file.path()).replace("\r\n", "\n"),
            "float32 (2, 3)\n[[0. 1. 2.]\n [3. 4. 5.]]\n"
        );
    }
}
