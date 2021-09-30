use crate::initlizer::initlizer::Initlizer;
use ndarray::{
    prelude::*,
    Array
};
use rand::distributions::{Distribution, Uniform};
extern crate num;

pub struct UniformInitlizer{
    high_range: f32,
    low_range: f32
}

impl UniformInitlizer{
    pub fn new(_high_range: f32, _low_range: f32) -> Self{
        Self {
            high_range: _high_range,
            low_range: _low_range
        }
    }
}

impl Initlizer<f32, 1> for UniformInitlizer {    
    fn get(&self, _shape: &[usize; 1]) -> Array<f32, Dim<[usize; 1]>>{
        let uni = Uniform::from(self.low_range..self.high_range);
        let mut rng = rand::thread_rng();
        let mut mat = Array::zeros(_shape.f());
        mat.iter_mut().for_each(|x| *x = uni.sample(&mut rng));
        return mat;
    }
}

impl Initlizer<f32, 2> for UniformInitlizer {    
    fn get(&self, _shape: &[usize; 2]) -> Array<f32, Dim<[usize; 2]>>{
        let uni = Uniform::from(self.low_range..self.high_range);
        let mut rng = rand::thread_rng();
        let mut mat = Array::zeros(_shape.f());
        mat.iter_mut().for_each(|x| *x = uni.sample(&mut rng));
        return mat;
    }
}

impl Initlizer<f32, 3> for UniformInitlizer {    
    fn get(&self, _shape: &[usize; 3]) -> Array<f32, Dim<[usize; 3]>>{
        let uni = Uniform::from(self.low_range..self.high_range);
        let mut rng = rand::thread_rng();
        let mut mat = Array::zeros(_shape.f());
        mat.iter_mut().for_each(|x| *x = uni.sample(&mut rng));
        return mat;
    }
}

impl Initlizer<f32, 4> for UniformInitlizer {    
    fn get(&self, _shape: &[usize; 4]) -> Array<f32, Dim<[usize; 4]>>{
        let uni = Uniform::from(self.low_range..self.high_range);
        let mut rng = rand::thread_rng();
        let mut mat = Array::zeros(_shape.f());
        mat.iter_mut().for_each(|x| *x = uni.sample(&mut rng));
        return mat;
    }
}
