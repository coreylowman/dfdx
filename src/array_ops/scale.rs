pub trait ScaleElements: Sized + Copy {
    fn scale(&self, rhs: &f32) -> Self {
        let mut result = *self;
        result.scale_assign(rhs);
        result
    }
    fn scale_assign(&mut self, rhs: &f32);
}

impl ScaleElements for f32 {
    fn scale_assign(&mut self, rhs: &f32) {
        *self *= rhs;
    }
}

impl<T: ScaleElements, const M: usize> ScaleElements for [T; M] {
    fn scale_assign(&mut self, rhs: &f32) {
        for i in 0..M {
            self[i].scale_assign(rhs);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale() {
        todo!("");
    }
}
