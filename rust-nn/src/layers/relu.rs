use ndarray::{
    prelude::*,
    Array,
    Zip
};

use crate::layers::layer::Layer;

pub struct ReLU<const M: usize> {
    pub layer_name: String,

    input: Array<f32, Dim<[usize; M]>>,
}

impl ReLU<1> {
    #[allow(dead_code)]
    pub fn new(_layer_name: &str) -> Self{
        Self {
            layer_name : _layer_name.to_string(),
            input: Array::<f32, _>::zeros([0].f())
        }
    }
}

impl ReLU<2> {
    #[allow(dead_code)]
    pub fn new(_layer_name: &str) -> Self{
        Self {
            layer_name : _layer_name.to_string(),
            input: Array::<f32, _>::zeros([0,0].f())
        }
    }
}

impl ReLU<3> {
    #[allow(dead_code)]
    pub fn new(_layer_name: &str) -> Self{
        Self {
            layer_name : _layer_name.to_string(),
            input: Array::<f32, _>::zeros([0,0,0].f())
        }
    }
}

impl ReLU<4> {
    #[allow(dead_code)]
    pub fn new(_layer_name: &str) -> Self{
        Self {
            layer_name : _layer_name.to_string(),
            input: Array::<f32, _>::zeros([0,0,0,0].f())
        }
    }
}

impl Layer<Array<f32, Ix1>> for ReLU<1> {
    fn forward(&mut self, _tensor: Array<f32, Ix1>) -> Array<f32, Ix1>{
        self.input = _tensor.clone();
        return _tensor.mapv(|x| if x > 0.0 {x} else {0.0});
    }

    fn backward(&mut self, mut _grad: Array<f32, Ix1>) -> Array<f32, Ix1>{
        Zip::from(&self.input)
            .and(&mut _grad)
            .for_each(|a, b| {
                if a < &0.0 {
                    *b = 0.0;
                }
            });
        return _grad;
    }
}

impl Layer<Array<f32, Ix2>> for ReLU<2> {
    fn forward(&mut self, _tensor: Array<f32, Ix2>) -> Array<f32, Ix2>{
        self.input = _tensor.clone();
        return _tensor.mapv(|x| if x > 0.0 {x} else {0.0});
    }

    fn backward(&mut self, mut _grad: Array<f32, Ix2>) -> Array<f32, Ix2>{
        Zip::from(&self.input)
            .and(&mut _grad)
            .for_each(|a, b| {
                if a < &0.0 {
                    *b = 0.0;
                }
            });
        return _grad;
    }
}

impl Layer<Array<f32, Ix3>> for ReLU<3> {
    fn forward(&mut self, _tensor: Array<f32, Ix3>) -> Array<f32, Ix3>{
        self.input = _tensor.clone();
        return _tensor.mapv(|x| if x > 0.0 {x} else {0.0});
    }

    fn backward(&mut self, mut _grad: Array<f32, Ix3>) -> Array<f32, Ix3>{
        Zip::from(&self.input)
            .and(&mut _grad)
            .for_each(|a, b| {
                if a < &0.0 {
                    *b = 0.0;
                }
            });
        return _grad;
    }
}

impl Layer<Array<f32, Ix4>> for ReLU<4> {
    fn forward(&mut self, _tensor: Array<f32, Ix4>) -> Array<f32, Ix4>{
        self.input = _tensor.clone();
        return _tensor.mapv(|x| if x > 0.0 {x} else {0.0});
    }

    fn backward(&mut self, mut _grad: Array<f32, Ix4>) -> Array<f32, Ix4>{
        Zip::from(&self.input)
            .and(&mut _grad)
            .for_each(|a, b| {
                if a < &0.0 {
                    *b = 0.0;
                }
            });
        return _grad;
    }
}