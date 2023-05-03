#![allow(clippy::needless_range_loop)]

use crate::{shapes::Dim, tensor::Cpu};

use super::cpu_kernel::MatMulImpl;

impl MatMulImpl<half::f16> for Cpu {
    #[inline]
    fn matmul<M: Dim, K: Dim, N: Dim>(
        (m, k, n): (M, K, N),
        ap: *const half::f16,
        a_strides: [usize; 2],
        bp: *const half::f16,
        b_strides: [usize; 2],
        cp: *mut half::f16,
        c_strides: [usize; 2],
    ) {
        const CHUNK_SIZE: usize = 8;

        let mut a_chunk: [f32; CHUNK_SIZE] = Default::default();
        let mut b_chunk: [[f32; CHUNK_SIZE]; CHUNK_SIZE] = Default::default();
        let mut c_chunk: [f32; CHUNK_SIZE] = Default::default();
        for i_m in 0..m.size() {
            let a_m = unsafe { ap.add(a_strides[0] * i_m) };
            let c_m = unsafe { cp.add(c_strides[0] * i_m) };
            for i_n_base in (0..n.size()).step_by(CHUNK_SIZE) {
                let n_chunk_size = CHUNK_SIZE.min(n.size() - i_n_base);

                {
                    // load c into chunk
                    for i_chunk_n in 0..n_chunk_size {
                        let c_mn = unsafe { *c_m.add(c_strides[1] * (i_n_base + i_chunk_n)) };
                        c_chunk[i_chunk_n] = c_mn.into();
                    }
                    for i_chunk_n in n_chunk_size..CHUNK_SIZE {
                        c_chunk[i_chunk_n] = 0.0;
                    }
                }

                for i_k_base in (0..k.size()).step_by(CHUNK_SIZE) {
                    let k_chunk_size = CHUNK_SIZE.min(k.size() - i_k_base);

                    {
                        // load a & b into chunk
                        for i_chunk_k in 0..k_chunk_size {
                            let i_k = i_k_base + i_chunk_k;
                            let a_mk = unsafe { *a_m.add(a_strides[1] * i_k) };
                            let b_k = unsafe { bp.add(b_strides[0] * i_k) };

                            // load a
                            a_chunk[i_chunk_k] = a_mk.into();

                            // load b
                            for i_chunk_n in 0..n_chunk_size {
                                let i_n = i_n_base + i_chunk_n;
                                let b_kn = unsafe { *b_k.add(b_strides[1] * i_n) };
                                b_chunk[i_chunk_k][i_chunk_n] = b_kn.into();
                            }
                            for i_chunk_n in n_chunk_size..CHUNK_SIZE {
                                b_chunk[i_chunk_k][i_chunk_n] = 0.0;
                            }
                        }
                        for i_chunk_k in k_chunk_size..CHUNK_SIZE {
                            a_chunk[i_chunk_k] = 0.0;
                            for i_chunk_n in n_chunk_size..CHUNK_SIZE {
                                b_chunk[i_chunk_k][i_chunk_n] = 0.0;
                            }
                        }
                    }

                    // do the computation
                    for i_n_chunk in 0..CHUNK_SIZE {
                        for i_k_chunk in 0..CHUNK_SIZE {
                            c_chunk[i_n_chunk] +=
                                a_chunk[i_k_chunk] * b_chunk[i_k_chunk][i_n_chunk];
                        }
                    }
                }

                // store c chunk in memory
                for (i_chunk, &c_f32) in c_chunk.iter().enumerate().take(n_chunk_size) {
                    let c = half::f16::from_f32(c_f32);
                    unsafe { *c_m.add(c_strides[1] * (i_n_base + i_chunk)) = c };
                }
            }
        }
    }
}
