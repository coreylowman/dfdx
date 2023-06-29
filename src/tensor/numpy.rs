use crate::shapes::{Dtype, HasShape, Shape};

use super::{CopySlice, Tensor};

use std::{
    fs::File,
    io::{self, BufReader, BufWriter, Read, Seek, Write},
    path::Path,
    string::{String, ToString},
    vec::Vec,
};

use zip::result::{ZipError, ZipResult};

const MAGIC_NUMBER: &[u8] = b"\x93NUMPY";
const VERSION: &[u8] = &[1, 0];

impl<S: Shape, E: Dtype + NumpyDtype, D: CopySlice<E>, T> Tensor<S, E, D, T> {
    /// Writes `data` to a new file in a zip archive named `filename`.
    pub fn write_to_npz<W: Write + Seek>(
        &self,
        w: &mut zip::ZipWriter<W>,
        filename: String,
    ) -> ZipResult<()> {
        let buf = self.as_vec();
        write_to_npz(w, self.shape().concrete().as_ref(), &buf, filename)?;
        Ok(())
    }

    /// Reads `data` from a file already in a zip archive named `filename`.
    pub fn read_from_npz<R: Read + Seek>(
        &mut self,
        r: &mut zip::ZipArchive<R>,
        filename: String,
    ) -> Result<(), NpzError> {
        let buf = read_from_npz(r, self.shape().concrete().as_ref(), filename)?;
        self.copy_from(&buf);
        Ok(())
    }

    /// Attemps to load the data from a `.npy` file at `path`
    pub fn load_from_npy<P: AsRef<Path>>(&mut self, path: P) -> Result<(), NpyError> {
        let mut f = BufReader::new(File::open(path)?);
        let buf = read_from_npy(&mut f, self.shape().concrete().as_ref())?;
        self.copy_from(&buf);
        Ok(())
    }

    /// Saves the tensor to a `.npy` file located at `path`
    pub fn save_to_npy<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let mut f = BufWriter::new(File::create(path)?);
        let buf = self.as_vec();
        write_to_npy(&mut f, self.shape().concrete().as_ref(), &buf)
    }
}

pub(crate) fn write_to_npz<W: Write + Seek, E: Dtype + NumpyDtype>(
    w: &mut zip::ZipWriter<W>,
    shape: &[usize],
    data: &[E],
    mut filename: String,
) -> io::Result<()> {
    if !filename.ends_with(".npy") {
        filename.push_str(".npy");
    }
    w.start_file(filename, Default::default())?;
    write_to_npy(w, shape, data)
}

pub(crate) fn read_from_npz<R: Read + Seek, E: Dtype + NumpyDtype>(
    r: &mut zip::ZipArchive<R>,
    shape: &[usize],
    mut filename: String,
) -> Result<Vec<E>, NpyError> {
    if !filename.ends_with(".npy") {
        filename.push_str(".npy");
    }

    let mut f = r
        .by_name(&filename)
        .unwrap_or_else(|_| panic!("'{filename}' not found"));

    read_from_npy(&mut f, shape)
}

fn write_to_npy<W: Write, E: Dtype + NumpyDtype>(
    w: &mut W,
    shape: &[usize],
    data: &[E],
) -> io::Result<()> {
    let endian = Endian::Little;
    write_header::<W, E>(w, endian, shape)?;

    for v in data.iter() {
        v.write_endian(w, endian)?;
    }

    Ok(())
}

fn read_from_npy<R: Read, E: Dtype + NumpyDtype>(
    r: &mut R,
    shape: &[usize],
) -> Result<Vec<E>, NpyError> {
    let endian = read_header::<R, E>(r, shape)?;
    let numel = shape.iter().product::<usize>();
    let mut out = Vec::new();

    for _ in 0..numel {
        out.push(E::read_endian(r, endian)?);
    }

    Ok(out)
}

fn write_header<W: Write, E: NumpyDtype>(
    w: &mut W,
    endian: Endian,
    shape: &[usize],
) -> io::Result<()> {
    let shape_str = to_shape_str(shape);

    let mut header: Vec<u8> = Vec::new();
    write!(
        &mut header,
        "{{'descr': '{}{}', 'fortran_order': False, 'shape': ({}), }}",
        match endian {
            Endian::Big => '>',
            Endian::Little => '<',
            Endian::Native => '=',
        },
        E::NUMPY_DTYPE_STR,
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

fn read_header<R: Read, E: NumpyDtype>(r: &mut R, shape: &[usize]) -> Result<Endian, NpyError> {
    let mut magic = [0; 6];
    r.read_exact(&mut magic)?;
    if magic != MAGIC_NUMBER {
        return Err(NpyError::InvalidMagicNumber(magic));
    }

    let mut version = [0; 2];
    r.read_exact(&mut version)?;
    if version != VERSION {
        return Err(NpyError::InvalidVersion(version));
    }

    let mut header_len_bytes = [0; 2];
    r.read_exact(&mut header_len_bytes)?;
    let header_len = u16::from_le_bytes(header_len_bytes);

    let mut header: Vec<u8> = std::vec![0; header_len as usize];
    r.read_exact(&mut header)?;

    let mut i = 0;
    i = expect(&header, i, b"{'descr': '")?;

    let endian = match header[i] {
        b'>' => Endian::Big,
        b'<' => Endian::Little,
        b'=' => Endian::Native,
        _ => return Err(NpyError::InvalidAlignment),
    };
    i += 1;

    i = expect(&header, i, E::NUMPY_DTYPE_STR.as_bytes())?;
    i = expect(&header, i, b"', ")?;

    // fortran order
    i = expect(&header, i, b"'fortran_order': False, ")?;

    // shape
    i = expect(&header, i, b"'shape': (")?;
    let shape_str = to_shape_str(shape);
    i = expect(&header, i, shape_str.as_bytes())?;
    expect(&header, i, b"), }")?;

    Ok(endian)
}

fn expect(buf: &[u8], i: usize, chars: &[u8]) -> Result<usize, NpyError> {
    for (offset, &c) in chars.iter().enumerate() {
        if buf[i + offset] != c {
            let expected = chars.to_vec();
            let found = buf[i..i + offset + 1].to_vec();
            let expected_str = String::from_utf8(expected.clone())?;
            let found_str = String::from_utf8(found.clone())?;
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

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Endian {
    Big,
    Little,
    Native,
}

/// Represents the NumpyDtype as a const str value.
///
/// Values should match up to the [numpy documentation](https://numpy.org/doc/stable/reference/arrays.dtypes.html)
/// for dtypes.
///
/// For example an f32's dtype is "f4".
pub trait NumpyDtype: Sized {
    const NUMPY_DTYPE_STR: &'static str;
    fn read_endian<R: Read>(r: &mut R, endian: Endian) -> io::Result<Self>;
    fn write_endian<W: Write>(&self, w: &mut W, endian: Endian) -> io::Result<()>;
}

impl NumpyDtype for f32 {
    const NUMPY_DTYPE_STR: &'static str = "f4";
    fn read_endian<R: Read>(r: &mut R, endian: Endian) -> io::Result<Self> {
        let mut bytes = [0; 4];
        r.read_exact(&mut bytes)?;
        Ok(match endian {
            Endian::Big => Self::from_be_bytes(bytes),
            Endian::Little => Self::from_le_bytes(bytes),
            Endian::Native => Self::from_ne_bytes(bytes),
        })
    }
    fn write_endian<W: Write>(&self, w: &mut W, endian: Endian) -> io::Result<()> {
        match endian {
            Endian::Big => w.write_all(&self.to_be_bytes()),
            Endian::Little => w.write_all(&self.to_le_bytes()),
            Endian::Native => w.write_all(&self.to_ne_bytes()),
        }
    }
}

impl NumpyDtype for f64 {
    const NUMPY_DTYPE_STR: &'static str = "f8";
    fn read_endian<R: Read>(r: &mut R, endian: Endian) -> io::Result<Self> {
        let mut bytes = [0; 8];
        r.read_exact(&mut bytes)?;
        Ok(match endian {
            Endian::Big => Self::from_be_bytes(bytes),
            Endian::Little => Self::from_le_bytes(bytes),
            Endian::Native => Self::from_ne_bytes(bytes),
        })
    }
    fn write_endian<W: Write>(&self, w: &mut W, endian: Endian) -> io::Result<()> {
        match endian {
            Endian::Big => w.write_all(&self.to_be_bytes()),
            Endian::Little => w.write_all(&self.to_le_bytes()),
            Endian::Native => w.write_all(&self.to_ne_bytes()),
        }
    }
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
    Utf8Error(std::string::FromUtf8Error),

    ParsingMismatch {
        expected: Vec<u8>,
        found: Vec<u8>,
        expected_str: String,
        found_str: String,
    },

    /// Unexpected alignment for [Endian].
    InvalidAlignment,
}

impl std::fmt::Display for NpyError {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            NpyError::InvalidMagicNumber(num) => write!(fmt, "invalid magic number: {num:?}"),
            NpyError::InvalidVersion(ver) => write!(fmt, "invalid version: {ver:?}"),
            NpyError::IoError(err) => write!(fmt, "{err}"),
            NpyError::Utf8Error(err) => write!(fmt, "{err}"),
            NpyError::ParsingMismatch {
                expected_str,
                found_str,
                ..
            } => write!(
                fmt,
                "error while parsing: expected {expected_str} found {found_str}"
            ),
            NpyError::InvalidAlignment => write!(fmt, "invalid alignment"),
        }
    }
}

impl std::error::Error for NpyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            NpyError::IoError(err) => Some(err),
            NpyError::Utf8Error(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for NpyError {
    fn from(e: std::io::Error) -> Self {
        Self::IoError(e)
    }
}

impl From<std::string::FromUtf8Error> for NpyError {
    fn from(e: std::string::FromUtf8Error) -> Self {
        Self::Utf8Error(e)
    }
}

fn to_shape_str(shape: &[usize]) -> String {
    shape
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<String>>()
        .join(", ")
        + if shape.len() == 1 { "," } else { "" }
}

/// Error that can happen while loading data from a `.npz` zip archive.
#[derive(Debug)]
pub enum NpzError {
    /// Something went wrong with reading from the `.zip` archive.
    Zip(ZipError),

    /// Something went wrong with loading data from a `.npy` file
    Npy(NpyError),
}

impl std::fmt::Display for NpzError {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            NpzError::Zip(err) => write!(fmt, "{err}"),
            NpzError::Npy(err) => write!(fmt, "{err}"),
        }
    }
}

impl std::error::Error for NpzError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            NpzError::Zip(err) => Some(err),
            NpzError::Npy(err) => Some(err),
        }
    }
}

impl From<NpyError> for NpzError {
    fn from(e: NpyError) -> Self {
        Self::Npy(e)
    }
}

impl From<ZipError> for NpzError {
    fn from(e: ZipError) -> Self {
        Self::Zip(e)
    }
}

impl From<std::io::Error> for NpzError {
    fn from(e: std::io::Error) -> Self {
        Self::Npy(e.into())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        tensor::{AsArray, TensorFrom},
        tests::TestDevice,
    };

    use super::*;

    use std::io::Read;
    use tempfile::NamedTempFile;

    #[test]
    fn test_0d_f32_save() {
        let dev: TestDevice = Default::default();

        let x = dev.tensor(0.0f32);

        let file = NamedTempFile::new().expect("failed to create tempfile");

        x.save_to_npy(file.path()).expect("Saving failed");

        let mut f = File::open(file.path()).expect("No file found");

        let mut found = Vec::new();
        f.read_to_end(&mut found).expect("Reading failed");

        assert_eq!(
            &found,
            &[
                147, 78, 85, 77, 80, 89, 1, 0, 64, 0, 123, 39, 100, 101, 115, 99, 114, 39, 58, 32,
                39, 60, 102, 52, 39, 44, 32, 39, 102, 111, 114, 116, 114, 97, 110, 95, 111, 114,
                100, 101, 114, 39, 58, 32, 70, 97, 108, 115, 101, 44, 32, 39, 115, 104, 97, 112,
                101, 39, 58, 32, 40, 41, 44, 32, 125, 32, 32, 32, 32, 32, 32, 32, 32, 10, 0, 0, 0,
                0,
            ]
        );
    }

    #[test]
    fn test_1d_f32_save() {
        let dev: TestDevice = Default::default();

        let x = dev.tensor([0.0f32, 1.0, 2.0, 3.0, -4.0]);

        let file = NamedTempFile::new().expect("failed to create tempfile");

        x.save_to_npy(file.path()).expect("Saving failed");

        let mut f = File::open(file.path()).expect("No file found");

        let mut found = Vec::new();
        f.read_to_end(&mut found).expect("Reading failed");

        assert_eq!(
            &found,
            &[
                147, 78, 85, 77, 80, 89, 1, 0, 64, 0, 123, 39, 100, 101, 115, 99, 114, 39, 58, 32,
                39, 60, 102, 52, 39, 44, 32, 39, 102, 111, 114, 116, 114, 97, 110, 95, 111, 114,
                100, 101, 114, 39, 58, 32, 70, 97, 108, 115, 101, 44, 32, 39, 115, 104, 97, 112,
                101, 39, 58, 32, 40, 53, 44, 41, 44, 32, 125, 32, 32, 32, 32, 32, 32, 10, 0, 0, 0,
                0, 0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 192
            ]
        );
    }

    #[test]
    fn test_2d_f32_save() {
        let dev: TestDevice = Default::default();

        let x = dev.tensor([[0.0f32, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        let file = NamedTempFile::new().expect("failed to create tempfile");

        x.save_to_npy(file.path()).expect("Saving failed");

        let mut f = File::open(file.path()).expect("No file found");

        let mut found = Vec::new();
        f.read_to_end(&mut found).expect("Reading failed");

        assert_eq!(
            &found,
            &[
                147, 78, 85, 77, 80, 89, 1, 0, 64, 0, 123, 39, 100, 101, 115, 99, 114, 39, 58, 32,
                39, 60, 102, 52, 39, 44, 32, 39, 102, 111, 114, 116, 114, 97, 110, 95, 111, 114,
                100, 101, 114, 39, 58, 32, 70, 97, 108, 115, 101, 44, 32, 39, 115, 104, 97, 112,
                101, 39, 58, 32, 40, 50, 44, 32, 51, 41, 44, 32, 125, 32, 32, 32, 32, 10, 0, 0, 0,
                0, 0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64, 0, 0, 160, 64
            ]
        );
    }

    #[test]
    fn test_0d_f32_load() {
        let dev: TestDevice = Default::default();
        let x = dev.tensor(2.0f32);

        let file = NamedTempFile::new().expect("failed to create tempfile");

        x.save_to_npy(file.path()).expect("Saving failed");

        let mut v = dev.tensor(0.0f32);
        v.load_from_npy(file.path()).expect("Loading failed");
        assert_eq!(v.array(), x.array());

        dev.tensor(0.0f64).load_from_npy(file.path()).expect_err("");
        dev.tensor([0.0f32; 1])
            .load_from_npy(file.path())
            .expect_err("");
    }

    #[test]
    fn test_1d_f32_load() {
        let dev: TestDevice = Default::default();

        let x = dev.tensor([0.0f32, 1.0, 2.0, 3.0, -4.0]);

        let file = NamedTempFile::new().expect("failed to create tempfile");

        x.save_to_npy(file.path()).expect("Saving failed");

        let mut value = dev.tensor([0.0f32; 5]);
        value.load_from_npy(file.path()).expect("");
        assert_eq!(value.array(), x.array());

        dev.tensor([0.0f64; 5])
            .load_from_npy(file.path())
            .expect_err("");
        dev.tensor(0.0f32).load_from_npy(file.path()).expect_err("");
        dev.tensor([[0.0f32; 2]; 3])
            .load_from_npy(file.path())
            .expect_err("");
    }

    #[test]
    fn test_2d_f32_load() {
        let dev: TestDevice = Default::default();
        let x = dev.tensor([[0.0f32, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        let file = NamedTempFile::new().expect("failed to create tempfile");

        x.save_to_npy(file.path()).expect("Saving failed");

        let mut value = dev.tensor([[0.0f32; 3]; 2]);
        assert!(value.load_from_npy(file.path()).is_ok());
        assert_eq!(value.array(), x.array());

        dev.tensor([0.0f64; 5])
            .load_from_npy(file.path())
            .expect_err("");
        dev.tensor(0.0f32).load_from_npy(file.path()).expect_err("");
        dev.tensor([[0.0f32; 2]; 3])
            .load_from_npy(file.path())
            .expect_err("");
    }
}
