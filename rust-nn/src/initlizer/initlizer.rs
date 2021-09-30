use ndarray::{
    prelude::*,
    Array
};

pub trait Initlizer<T, const D: usize> {
    fn get(&self, _shape: &[usize; D]) -> Array<T, Dim<[usize; D]>>;
}