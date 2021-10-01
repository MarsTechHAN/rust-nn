mod layers;
mod initlizer;
mod helper;

use layers::layer::Layer;
use layers::dense::Dense;
use layers::relu::ReLU;
use initlizer::uniform_initlizer::UniformInitlizer;
use initlizer::const_initlizer::ConstInitlizer;
use helper::math::*;

use ndarray::{
    Array
};

fn main() {
    // _layer_name: String, _num_in: usize, _num_out: usize, _w_initlizer: W, _b_initlizer: B
    let mut in_layer = Dense::new("IN", 10, 5, UniformInitlizer::new(1.0, -1.0), ConstInitlizer::new(0.0));
    let mut relu_layer = ReLU::<2>::new("RELU");

    let in_mat = Array::linspace(-10., 10., 10).into_shape((1,10)).unwrap();
    println!("Input: {:?}", in_mat);
    let out_mat = in_layer.forward(in_mat);
    println!("Out: {:?}", out_mat);
    let out_act_mat = relu_layer.forward(out_mat);
    println!("Relu: {:?}", out_act_mat);

    println!("Softmax: {:?}", Softmax::<2>::softmax(out_act_mat.clone(), 1.0, -1));
    println!("LogSoftmax: {:?}", LogSoftmax::<2>::log_softmax(out_act_mat.clone(), 1.0, -1));
    println!("Sigmoid: {:?}", Sigmoid::<2>::sigmoid(out_act_mat.clone()));
}