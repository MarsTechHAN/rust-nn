use crate::initlizer::initlizer::Initlizer;
use ndarray::{
    prelude::*,
    Array
};
extern crate num;

pub struct ZeroInitlizer{
    
}

impl ZeroInitlizer{
    pub fn new() -> Self {
        Self{}
    }
}

impl Initlizer<f32, 1> for ZeroInitlizer {    
    fn get(&self, _shape: &[usize; 1]) -> Array<f32, Dim<[usize; 1]>>{
        return Array::zeros(_shape.f());
    }
}

impl Initlizer<f32, 2> for ZeroInitlizer {    
    fn get(&self, _shape: &[usize; 2]) -> Array<f32, Dim<[usize; 2]>>{
        return Array::<f32, _>::zeros(_shape.f());
    }
}

impl Initlizer<f32, 3> for ZeroInitlizer {    
    fn get(&self, _shape: &[usize; 3]) -> Array<f32, Dim<[usize; 3]>>{
        return Array::zeros(_shape.f());
    }
}

impl Initlizer<f32, 4> for ZeroInitlizer {    
    fn get(&self, _shape: &[usize; 4]) -> Array<f32, Dim<[usize; 4]>>{
        return Array::zeros(_shape.f());
    }
}
