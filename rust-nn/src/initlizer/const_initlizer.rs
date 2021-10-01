use crate::initlizer::initlizer::Initlizer;
use ndarray::{
    prelude::*,
    Array
};
extern crate num;

pub struct ConstInitlizer{
    value: f32
}

impl ConstInitlizer{
    pub fn new(_value: f32) -> Self {
        Self{
            value: _value
        }
    }
}

impl Initlizer<f32, 1> for ConstInitlizer {    
    fn get(&self, _shape: &[usize; 1]) -> Array<f32, Dim<[usize; 1]>>{
        let mut mat = Array::zeros(_shape.f());
        mat.fill(self.value);
        return mat;
    }
}

impl Initlizer<f32, 2> for ConstInitlizer {    
    fn get(&self, _shape: &[usize; 2]) -> Array<f32, Dim<[usize; 2]>>{
        let mut mat = Array::zeros(_shape.f());
        mat.fill(self.value);
        return mat;
    }
}

impl Initlizer<f32, 3> for ConstInitlizer {    
    fn get(&self, _shape: &[usize; 3]) -> Array<f32, Dim<[usize; 3]>>{
        return Array::zeros(_shape.f());
    }
}

impl Initlizer<f32, 4> for ConstInitlizer {    
    fn get(&self, _shape: &[usize; 4]) -> Array<f32, Dim<[usize; 4]>>{
        return Array::zeros(_shape.f());
    }
}
