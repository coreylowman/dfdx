//! Provides some generic functions to save Nd arrays in the .npy format.

use super::*;
use std::{
    fs::File,
    io::{BufReader, Read},
    path::Path,
    string::FromUtf8Error,
};

/// Loads data from a .npy file. This calls [read()].
///
/// This is implemented for an arbitrarily shaped array.
/// See [ReadNumbers] for how this is done (recursive array traits!).
///
/// Currently only implemented for f32 and f64 arrays. To add another
/// base type, you can implement [NumpyShape]
///
/// Example Usage:
/// ```ignore
/// use dfdx::numpy;
/// let mut arr = [[0.0f32; 3]; 2];
/// numpy::load("test.npy", &mut arr);
/// ```
pub fn load<T, P>(path: P, t: &mut T) -> Result<(), NpyError>
where
    T: NumpyDtype + NumpyShape + ReadNumbers,
    P: AsRef<Path>,
{
    let mut f = BufReader::new(File::open(path).map_err(NpyError::IoError)?);
    read(&mut f, t)
}

/// Reads data from a [Read].
///
/// There are a lot of errors that can happen during this process. See [ReadError]
/// for more info.
///
/// The overall process is:
/// 1. Read the .npy header.
/// 2. Make sure T's [NumpyDtype::Dtype] matches the header's dtype
/// 3. Make sure T's [NumpyShape::shape()] matches the header's shape
/// 4. Parse an [Endian] from header's "descr" field.
/// 5. Read the data using [ReadNumbers].
///
/// Multiple errors can happen from each of the above parts!
pub fn read<T, R>(r: &mut R, t: &mut T) -> Result<(), NpyError>
where
    T: NumpyDtype + NumpyShape + ReadNumbers,
    R: Read,
{
    let endian = read_header::<T, R>(r)?;
    t.read_numbers(r, endian).map_err(NpyError::IoError)?;
    Ok(())
}

#[derive(Debug)]
pub enum NpyError {
    /// Magic number did not match the expected value.
    InvalidMagicNumber([u8; 6]),

    // Version did not match the expected value.
    InvalidVersion([u8; 2]),

    /// Error from opening a file, reading values, etc.
    IoError(std::io::Error),

    /// Error from converting header bytes to a [String].
    Utf8Error(FromUtf8Error),

    ParsingMismatch {
        expected: Vec<u8>,
        found: Vec<u8>,
        expected_str: String,
        found_str: String,
    },

    /// Unexpected alignment for [Endian].
    InvalidAlignment,
}

fn read_header<T, R>(r: &mut R) -> Result<Endian, NpyError>
where
    T: NumpyDtype + NumpyShape,
    R: Read,
{
    let mut magic = [0; 6];
    r.read_exact(&mut magic).map_err(NpyError::IoError)?;
    if magic != MAGIC_NUMBER {
        return Err(NpyError::InvalidMagicNumber(magic));
    }

    let mut version = [0; 2];
    r.read_exact(&mut version).map_err(NpyError::IoError)?;
    if version != VERSION {
        return Err(NpyError::InvalidVersion(version));
    }

    let mut header_len_bytes = [0; 2];
    r.read_exact(&mut header_len_bytes)
        .map_err(NpyError::IoError)?;
    let header_len = u16::from_le_bytes(header_len_bytes);

    let mut header: Vec<u8> = vec![0; header_len as usize];
    r.read_exact(&mut header).map_err(NpyError::IoError)?;

    let mut i = 0;
    i = expect(&header, i, b"{'descr': '")?;

    let endian = match header[i] {
        b'>' => Endian::Big,
        b'<' => Endian::Little,
        b'=' => Endian::Native,
        _ => return Err(NpyError::InvalidAlignment),
    };
    i += 1;

    i = expect(&header, i, T::DTYPE.as_bytes())?;
    i = expect(&header, i, b"', ")?;

    // fortran order
    i = expect(&header, i, b"'fortran_order': False, ")?;

    // shape
    i = expect(&header, i, b"'shape': (")?;
    let shape_str = to_shape_str(T::shape());
    i = expect(&header, i, shape_str.as_bytes())?;
    expect(&header, i, b"), }")?;

    Ok(endian)
}

fn expect(buf: &[u8], i: usize, chars: &[u8]) -> Result<usize, NpyError> {
    for (offset, &c) in chars.iter().enumerate() {
        if buf[i + offset] != c {
            let expected = chars.to_vec();
            let found = buf[i..i + offset + 1].to_vec();
            let expected_str = String::from_utf8(expected.clone()).map_err(NpyError::Utf8Error)?;
            let found_str = String::from_utf8(found.clone()).map_err(NpyError::Utf8Error)?;
            return Err(NpyError::ParsingMismatch {
                expected,
                found,
                expected_str,
                found_str,
            });
        }
    }
    Ok(i + chars.len())
}

/// Reads all the numbers from `r` into `&mut self` assuming the bytes are layed out in [Endian] order.
/// Most types that this should be implemented for have `Self::from_be_bytes()`, `Self::from_le_bytes()`,
/// and `Self::from_ne_bytes()`.
pub trait ReadNumbers {
    fn read_numbers<R: Read>(&mut self, r: &mut R, endian: Endian) -> std::io::Result<()>;
}

impl ReadNumbers for f32 {
    fn read_numbers<R: Read>(&mut self, r: &mut R, endian: Endian) -> std::io::Result<()> {
        let mut bytes = [0; 4];
        r.read_exact(&mut bytes)?;
        *self = match endian {
            Endian::Big => Self::from_be_bytes(bytes),
            Endian::Little => Self::from_le_bytes(bytes),
            Endian::Native => Self::from_ne_bytes(bytes),
        };
        Ok(())
    }
}

impl ReadNumbers for f64 {
    fn read_numbers<R: Read>(&mut self, r: &mut R, endian: Endian) -> std::io::Result<()> {
        let mut bytes = [0; 8];
        r.read_exact(&mut bytes)?;
        *self = match endian {
            Endian::Big => Self::from_be_bytes(bytes),
            Endian::Little => Self::from_le_bytes(bytes),
            Endian::Native => Self::from_ne_bytes(bytes),
        };
        Ok(())
    }
}

impl<T: ReadNumbers, const M: usize> ReadNumbers for [T; M] {
    fn read_numbers<R: Read>(&mut self, r: &mut R, endian: Endian) -> std::io::Result<()> {
        for self_i in self.iter_mut() {
            self_i.read_numbers(r, endian)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_0d_f32_load() {
        let data: f32 = 2.0;

        let file = NamedTempFile::new().expect("failed to create tempfile");

        save(file.path(), &data).expect("Saving failed");

        let mut v = 0.0f32;
        load(file.path(), &mut v).expect("");
        assert_eq!(v, data);

        let mut v = 0.0f64;
        load(file.path(), &mut v).expect_err("");

        let mut v = [0.0f32; 1];
        load(file.path(), &mut v).expect_err("");
    }

    #[test]
    fn test_1d_f32_save() {
        let data: [f32; 5] = [0.0, 1.0, 2.0, 3.0, -4.0];

        let file = NamedTempFile::new().expect("failed to create tempfile");

        save(file.path(), &data).expect("Saving failed");

        let mut value = [0.0f32; 5];
        load(file.path(), &mut value).expect("");
        assert_eq!(value, data);

        let mut value = [0.0f64; 5];
        load(file.path(), &mut value).expect_err("");

        let mut value = 0.0f32;
        load(file.path(), &mut value).expect_err("");

        let mut value = [[0.0f32; 2]; 3];
        load(file.path(), &mut value).expect_err("");
    }

    #[test]
    fn test_2d_f32_save() {
        let data: [[f32; 3]; 2] = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]];

        let file = NamedTempFile::new().expect("failed to create tempfile");

        save(file.path(), &data).expect("Saving failed");

        let mut value = [[0.0f32; 3]; 2];
        assert!(load(file.path(), &mut value).is_ok());
        assert_eq!(value, data);

        let mut value = [0.0f64; 5];
        assert!(load(file.path(), &mut value).is_err());

        let mut value = 0.0f32;
        assert!(load(file.path(), &mut value).is_err());

        let mut value = [[0.0f32; 2]; 3];
        assert!(load(file.path(), &mut value).is_err());
    }
}
