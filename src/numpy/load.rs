//! Provides some generic functions to save Nd arrays in the .npy format.

use super::*;
use num_bigint::{BigInt, BigUint};
use std::{
    fs::File,
    io::{BufReader, Read, Result},
    path::Path,
    str::Utf8Error,
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
/// ```no_run
/// use dfdx::numpy;
/// let mut arr = [[0.0f32; 3]; 2];
/// numpy::load("test.npy", &mut arr);
/// ```
pub fn load<T, P>(path: P, t: &mut T) -> std::result::Result<(), ReadError>
where
    T: NumpyDtype + NumpyShape + ReadNumbers,
    P: AsRef<Path>,
{
    let mut f = BufReader::new(File::open(path).map_err(ReadError::IoError)?);
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
pub fn read<T, R>(r: &mut R, t: &mut T) -> std::result::Result<(), ReadError>
where
    T: NumpyDtype + NumpyShape + ReadNumbers,
    R: Read,
{
    let header = read_header(r)?;
    let expected_shape = T::shape();
    if expected_shape.len() != header.shape.len() {
        return Err(ReadError::WrongShape);
    }
    for (&e, f) in T::shape().iter().zip(header.shape.iter()) {
        if &BigInt::from(BigUint::from(e)) != f {
            return Err(ReadError::WrongShape);
        }
    }
    let endian = match &header.descr.chars().nth(0) {
        Some('>') => Endian::Big,
        Some('<') => Endian::Little,
        Some('=') => Endian::Native,
        _ => return Err(ReadError::InvalidAlignment),
    };

    if T::DTYPE != &header.descr[1..] {
        return Err(ReadError::WrongDtype);
    }

    t.read_numbers(r, endian).map_err(ReadError::IoError)?;

    Ok(())
}

#[derive(Debug)]
pub enum ReadError {
    /// Magic number did not match the expected value.
    InvalidMagicNumber([u8; 6]),

    // Version did not match the expected value.
    InvalidVersion([u8; 2]),

    /// Error from opening a file, reading values, etc.
    IoError(std::io::Error),

    /// Error from converting header bytes to a [String].
    Utf8Error(Utf8Error),

    /// Error from convert header [String] into a [py_literal::Value].
    PyLiteral(py_literal::ParseError),

    /// The header is not a python dictionary.
    HeaderNotADict,

    /// The header dictionary is missing the "descr" key.
    HeaderMissingDescr,

    /// The header dictionary is missing the "fortran_order" key.
    HeaderMissingFortranOrder,

    /// The header dictionary is missing the "shape" key.
    HeaderMissingShape,

    /// The header dictionary value for "descr" is invalid in some way.
    HeaderInvalidDescr,

    /// The header dictionary value for "fortran_order" is invalid in some way.
    HeaderInvalidFortranOrder,

    /// The header dictionary value for "shape" is invalid in some way.
    HeaderInvalidShape,

    /// The shape from the header is not what was expected.
    WrongShape,

    /// Unexpected alignment for [Endian].
    InvalidAlignment,

    /// The dtype from the header is not what was expected.
    WrongDtype,
}

struct ParsedHeader {
    descr: String,
    shape: Vec<BigInt>,
}

fn read_header<R>(r: &mut R) -> std::result::Result<ParsedHeader, ReadError>
where
    R: Read,
{
    let mut magic = [0; 6];
    r.read_exact(&mut magic).map_err(ReadError::IoError)?;
    if magic != MAGIC_NUMBER {
        return Err(ReadError::InvalidMagicNumber(magic));
    }

    let mut version = [0; 2];
    r.read_exact(&mut version).map_err(ReadError::IoError)?;
    if version != VERSION {
        return Err(ReadError::InvalidVersion(version));
    }

    let mut header_len_bytes = [0; 2];
    r.read_exact(&mut header_len_bytes)
        .map_err(ReadError::IoError)?;
    let header_len = u16::from_le_bytes(header_len_bytes);

    let mut header_bytes: Vec<u8> = vec![0; header_len as usize];
    r.read_exact(&mut header_bytes)
        .map_err(ReadError::IoError)?;

    let header = std::str::from_utf8(&header_bytes)
        .map_err(ReadError::Utf8Error)?
        .to_string();

    let py_header: py_literal::Value = header.trim().parse().map_err(ReadError::PyLiteral)?;
    let items = py_header.as_dict().ok_or(ReadError::HeaderNotADict)?;

    let mut descr = None;
    let mut shape = None;
    let mut fortran_order = None;
    for (key, value) in items.iter() {
        if key == &py_literal::Value::String("descr".into()) {
            descr = Some(value);
        }
        if key == &py_literal::Value::String("shape".into()) {
            shape = Some(value);
        }
        if key == &py_literal::Value::String("fortran_order".into()) {
            fortran_order = Some(value);
        }
    }

    let descr = descr.ok_or(ReadError::HeaderMissingDescr)?;
    let fortran_order = fortran_order.ok_or(ReadError::HeaderMissingFortranOrder)?;
    let shape = shape.ok_or(ReadError::HeaderInvalidShape)?;

    if fortran_order != &py_literal::Value::Boolean(false) {
        return Err(ReadError::HeaderInvalidFortranOrder);
    }

    let descr = descr
        .as_string()
        .ok_or(ReadError::HeaderInvalidDescr)?
        .clone();

    let shape_values = shape.as_tuple().ok_or(ReadError::HeaderInvalidShape)?;

    let mut shape: Vec<BigInt> = Vec::new();
    for item in shape_values.iter() {
        let v = match item {
            py_literal::Value::Integer(value) => value.clone(),
            _ => return Err(ReadError::HeaderInvalidShape),
        };
        shape.push(v);
    }

    Ok(ParsedHeader { descr, shape })
}

/// Reads all the numbers from `r` into `&mut self` assuming the bytes are layed out in [Endian] order.
/// Most types that this should be implemented for have `Self::from_be_bytes()`, `Self::from_le_bytes()`,
/// and `Self::from_ne_bytes()`.
pub trait ReadNumbers {
    fn read_numbers<R: Read>(&mut self, r: &mut R, endian: Endian) -> Result<()>;
}

impl ReadNumbers for f32 {
    fn read_numbers<R: Read>(&mut self, r: &mut R, endian: Endian) -> Result<()> {
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
    fn read_numbers<R: Read>(&mut self, r: &mut R, endian: Endian) -> Result<()> {
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
    fn read_numbers<R: Read>(&mut self, r: &mut R, endian: Endian) -> Result<()> {
        for i in 0..M {
            self[i].read_numbers(r, endian)?;
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
        let data: f32 = 3.14;

        let file = NamedTempFile::new().expect("failed to create tempfile");

        save(file.path(), &data).expect("Saving failed");

        let mut v = 0.0f32;
        assert!(load(file.path(), &mut v).is_ok());
        assert_eq!(v, data);

        let mut v = 0.0f64;
        assert!(load(file.path(), &mut v).is_err());

        let mut v = [0.0f32; 1];
        assert!(load(file.path(), &mut v).is_err());
    }

    #[test]
    fn test_1d_f32_save() {
        let data: [f32; 5] = [0.0, 1.0, 2.0, 3.0, -4.0];

        let file = NamedTempFile::new().expect("failed to create tempfile");

        save(file.path(), &data).expect("Saving failed");

        let mut value = [0.0f32; 5];
        assert!(load(file.path(), &mut value).is_ok());
        assert_eq!(value, data);

        let mut value = [0.0f64; 5];
        assert!(load(file.path(), &mut value).is_err());

        let mut value = 0.0f32;
        assert!(load(file.path(), &mut value).is_err());

        let mut value = [[0.0f32; 2]; 3];
        assert!(load(file.path(), &mut value).is_err());
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
