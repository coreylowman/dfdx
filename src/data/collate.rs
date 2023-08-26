use std::{mem::MaybeUninit, vec::Vec};

/// Collates `Self` into some other type.
/// Generally similar to an unzip method;
/// Transforms `[(A, B, ...); N]` into `([A; N], [B; N], ...)`;
/// Transforms `Vec<(A, B, ...)>` into `(Vec<A>, Vec<B>, ...)`.
pub trait Collate {
    type Collated;
    fn collated(self) -> Self::Collated;
}

macro_rules! generate_collate_impl {
    ( $( [$generic:ident, $var:ident, $iter:ident] ),* ) => {
        // non-reference, array
        impl<$($generic,)* const N: usize> Collate for [($($generic,)*); N] {
            type Collated = ($([$generic; N]),*);
            fn collated(self) -> Self::Collated {
                $( let mut $var = [(); N].map(|_| MaybeUninit::uninit()); )*

                for (i, ($($iter,)*)) in self.into_iter().enumerate() {
                    $( $var[i].write($iter); )*
                }

                $( let $var = unsafe { $var.map(|$iter| $iter.assume_init()) }; )*

                ($($var,)*)
            }
        }

        // reference, array
        impl<'a, $($generic,)* const N: usize> Collate for [&'a ($($generic,)*); N] {
            type Collated = ($([&'a $generic; N]),*);
            fn collated(self) -> Self::Collated {
                $( let mut $var = [(); N].map(|_| MaybeUninit::uninit()); )*

                for (i, ($($iter,)*)) in self.into_iter().enumerate() {
                    $( $var[i].write($iter); )*
                }

                $( let $var = unsafe { $var.map(|$iter| $iter.assume_init()) }; )*

                ($($var,)*)
            }
        }

        // non-reference, vec
        impl <$($generic,)*> Collate for Vec<($($generic,)*)> {
            type Collated = ($(Vec<$generic>),*);
            fn collated(self) -> Self::Collated {
                $( let mut $var = Vec::with_capacity(self.len()); )*

                for ($($iter,)*) in self.into_iter() {
                    $( $var.push($iter); )*
                }

                ($($var,)*)
            }
        }

        // reference, vec
        impl <'a, $($generic,)*> Collate for Vec<&'a ($($generic,)*)> {
            type Collated = ($(Vec<&'a $generic>),*);
            fn collated(self) -> Self::Collated {
                $( let mut $var = Vec::with_capacity(self.len()); )*

                for ($($iter,)*) in self.into_iter() {
                    $( $var.push($iter); )*
                }

                ($($var,)*)
            }
        }
    };
}

generate_collate_impl!([A, a, a_n], [B, b, b_n]);
generate_collate_impl!([A, a, a_n], [B, b, b_n], [C, c, c_n]);
generate_collate_impl!([A, a, a_n], [B, b, b_n], [C, c, c_n], [D, d, d_n]);
generate_collate_impl!(
    [A, a, a_n],
    [B, b, b_n],
    [C, c, c_n],
    [D, d, d_n],
    [E, e, e_n]
);
generate_collate_impl!(
    [A, a, a_n],
    [B, b, b_n],
    [C, c, c_n],
    [D, d, d_n],
    [E, e, e_n],
    [F, f, f_n]
);
generate_collate_impl!(
    [A, a, a_n],
    [B, b, b_n],
    [C, c, c_n],
    [D, d, d_n],
    [E, e, e_n],
    [F, f, f_n],
    [G, g, g_n]
);
generate_collate_impl!(
    [A, a, a_n],
    [B, b, b_n],
    [C, c, c_n],
    [D, d, d_n],
    [E, e, e_n],
    [F, f, f_n],
    [G, g, g_n],
    [H, h, h_n]
);
generate_collate_impl!(
    [A, a, a_n],
    [B, b, b_n],
    [C, c, c_n],
    [D, d, d_n],
    [E, e, e_n],
    [F, f, f_n],
    [G, g, g_n],
    [H, h, h_n],
    [I, i, i_n]
);

pub struct Collator<I> {
    iter: I,
}

impl<I: Iterator> Iterator for Collator<I>
where
    I::Item: Collate,
{
    type Item = <I::Item as Collate>::Collated;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|i| i.collated())
    }
}

impl<I: ExactSizeIterator> ExactSizeIterator for Collator<I>
where
    Self: Iterator,
{
    fn len(&self) -> usize {
        self.iter.len()
    }
}

/// Collate an [Iterator] where Self::Item impls [Collate].
pub trait IteratorCollateExt: Iterator {
    /// Collates (unzips) items - very similar to [std::iter::Iterator::unzip],
    /// but works on items that are Vecs or arrays
    ///
    /// For example:
    /// - An item of `Vec<(usize, usize, ...)>` becomes `(Vec<usize>, Vec<usize>, ...)`
    /// - An item of `[(usize, usize, ...); N]` becomes `([usize; N], [usize; N], ...)`
    ///
    /// Example implementations:
    /// ```rust
    /// # use dfdx::data::IteratorCollateExt;
    /// let data = [[('a', 'b'); 10], [('c', 'd'); 10], [('e', 'f'); 10]];
    /// // we use collate to transform each batch:
    /// let mut iter = data.into_iter().collate();
    /// assert_eq!(iter.next().unwrap(), (['a'; 10], ['b'; 10]));
    /// assert_eq!(iter.next().unwrap(), (['c'; 10], ['d'; 10]));
    /// assert_eq!(iter.next().unwrap(), (['e'; 10], ['f'; 10]));
    /// ```
    ///
    /// This works for any number of items in the tuple up to 8.
    fn collate(self) -> Collator<Self>
    where
        Self: Sized,
    {
        Collator { iter: self }
    }
}
impl<I: Iterator> IteratorCollateExt for I {}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_collate_array() {
        let items = [(1, 2), (3, 4), (5, 6)];
        assert_eq!(items.collated(), ([1, 3, 5], [2, 4, 6]));

        let items = [&(1, 2), &(3, 4), &(5, 6)];
        assert_eq!(items.collated(), ([&1, &3, &5], [&2, &4, &6]));

        let items = [(1, 2, 3), (4, 5, 6), (7, 8, 9)];
        assert_eq!(items.collated(), ([1, 4, 7], [2, 5, 8], [3, 6, 9]));

        let items = [&(1, 2, 3), &(4, 5, 6), &(7, 8, 9)];
        assert_eq!(items.collated(), ([&1, &4, &7], [&2, &5, &8], [&3, &6, &9]));
    }

    #[test]
    fn test_collate_vec() {
        let items = std::vec![(1, 2), (3, 4), (5, 6)];
        let (a, b): (Vec<i32>, Vec<i32>) = items.collated();
        assert_eq!(a, [1, 3, 5]);
        assert_eq!(b, [2, 4, 6]);

        let items = std::vec![&(1, 2), &(3, 4), &(5, 6)];
        let (a, b): (Vec<&i32>, Vec<&i32>) = items.collated();
        assert_eq!(a, [&1, &3, &5]);
        assert_eq!(b, [&2, &4, &6]);

        let items = std::vec![(1, 2, 3), (4, 5, 6), (7, 8, 9)];
        let (a, b, c): (Vec<i32>, Vec<i32>, Vec<i32>) = items.collated();
        assert_eq!(a, [1, 4, 7]);
        assert_eq!(b, [2, 5, 8]);

        let items = std::vec![&(1, 2, 3), &(4, 5, 6), &(7, 8, 9)];
        let (a, b, c): (Vec<&i32>, Vec<&i32>, Vec<&i32>) = items.collated();
        assert_eq!(a, [&1, &4, &7]);
        assert_eq!(b, [&2, &5, &8]);
    }
}
