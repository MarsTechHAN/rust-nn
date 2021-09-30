use ndarray::{
    prelude::*,
    Array
};

use crate::layers::layer::Layer;
use crate::initlizer::initlizer::Initlizer;

pub struct Dense {
    pub layer_name: String,

    pub num_in: usize,
    pub num_out: usize,

    w: Array<f32, Ix2>,
    b: Array<f32, Ix1>,

    input: Array<f32, Ix2>,
    g_w: Array<f32, Ix2>,
    g_b: Array<f32, Ix1>,
}

impl Dense {
    pub fn new<W: Initlizer<f32, 2>, B: Initlizer<f32, 1>>(_layer_name: &str, _num_in: usize, _num_out: usize, _w_initlizer: W, _b_initlizer: B) -> Self{
        Self {
            layer_name : _layer_name.to_string(),
            num_in: _num_in,
            num_out: _num_out,
            w: _w_initlizer.get(&[_num_in, _num_out]),
            b: _b_initlizer.get(&[_num_out]),
            input: Array::<f32, Ix2>::zeros([1, _num_in].f()),
            g_w: Array::<f32, Ix2>::zeros([_num_in, _num_out].f()),
            g_b: Array::<f32, Ix1>::zeros([_num_out].f()),
        }
    }
}

impl Layer<Array<f32, Ix2>> for Dense {
    fn forward(&mut self, _tensor: Array<f32, Ix2>) -> Array<f32, Ix2>{
        self.input = _tensor.clone();
        return (_tensor.dot(&self.w)) + &self.b;
    }

    fn backward(&mut self, _grad: Array<f32, Ix2>) -> Array<f32, Ix2>{
        self.g_w = self.input.t().dot(&_grad);
        self.g_b = _grad.sum_axis(Axis(0));
        return _grad.dot(&self.w.t());
    }
}