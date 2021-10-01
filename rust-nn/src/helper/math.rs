use ndarray::{
    prelude::*,
    Array,
};
use std::cmp::Ordering;

pub struct Softmax<const M: usize> {
}

impl Softmax<1>{
    #[allow(dead_code)]
    pub fn softmax(_in: Array<f32, Dim<[usize; 1]>>, _t: f32) -> Array<f32, Dim<[usize; 1]>>{
        let in_dt = &_in / _t;
        let x_max = in_dt.iter().fold(-f32::INFINITY, |a, &b| {
            match PartialOrd::partial_cmp(&a, &b) {
                None => f32::NAN,
                Some(Ordering::Greater) => a,
                Some(_) => b,
            }
        });
        let mut exps = in_dt - x_max;
        exps.iter_mut().for_each(|x| *x = x.exp());
        return &exps / exps.sum();
    }
}

impl Softmax<2>{
    #[allow(dead_code)]
    pub fn softmax(_in: Array<f32, Dim<[usize; 2]>>, _t: f32, _dim: i8) -> Array<f32, Dim<[usize; 2]>>{
        let mut in_dt = &_in / _t;
        if _dim == -1 || _dim == 0 || _dim == 1{
            if _dim == 1{
                in_dt = in_dt.t().to_owned();
            }

            for row in 0..in_dt.shape()[0] {
                let x_max = in_dt.index_axis(Axis(0), row).iter().fold(-f32::INFINITY, |a, &b| {
                    match PartialOrd::partial_cmp(&a, &b) {
                        None => f32::NAN,
                        Some(Ordering::Greater) => a,
                        Some(_) => b,
                    }
                });
                in_dt.index_axis_mut(Axis(0), row).iter_mut().for_each(|x| *x = *x - x_max);
            }

            in_dt.iter_mut().for_each(|x| *x = x.exp());

            let out;
            if _dim == 1{
                out = (&in_dt / in_dt.sum()).t().to_owned();
            }else{
                out = &in_dt / in_dt.sum();
            }

            return out;
        }else{
            panic!("Not a valid dim for 2D ndarry.");
        }
    }
}

pub struct LogSoftmax<const M: usize> {
}

impl LogSoftmax<1>{
    #[allow(dead_code)]
    pub fn log_softmax(_in: Array<f32, Dim<[usize; 1]>>, _t: f32) -> Array<f32, Dim<[usize; 1]>>{
        let in_dt = &_in / _t;
        let x_max = in_dt.iter().fold(-f32::INFINITY, |a, &b| {
            match PartialOrd::partial_cmp(&a, &b) {
                None => f32::NAN,
                Some(Ordering::Greater) => a,
                Some(_) => b,
            }
        });
        let mut exps = &in_dt - x_max;
        exps.iter_mut().for_each(|x| *x = x.exp());
        let exps_sum = exps.sum().ln();
        return in_dt - x_max - exps_sum;
    }
}

impl LogSoftmax<2>{
    #[allow(dead_code)]
    pub fn log_softmax(_in: Array<f32, Dim<[usize; 2]>>, _t: f32, _dim: i8) -> Array<f32, Dim<[usize; 2]>>{
        let mut in_dt = &_in / _t;
        if _dim == -1 || _dim == 0 || _dim == 1{
            if _dim == 1{
                in_dt = in_dt.t().to_owned();
            }
            
            let mut _out = in_dt.clone();

            for row in 0..in_dt.shape()[0] {
                let x_max = in_dt.index_axis(Axis(0), row).iter().fold(-f32::INFINITY, |a, &b| {
                    match PartialOrd::partial_cmp(&a, &b) {
                        None => f32::NAN,
                        Some(Ordering::Greater) => a,
                        Some(_) => b,
                    }
                });
                _out.index_axis_mut(Axis(0), row).iter_mut().for_each(|x| *x = *x - x_max);
                _out.index_axis_mut(Axis(0), row).iter_mut().for_each(|x| *x = x.exp());
                let exps_sum = _out.index_axis_mut(Axis(0), row).sum().ln();
                in_dt.index_axis_mut(Axis(0), row).iter_mut().for_each(|x| *x = *x - x_max - exps_sum);
            }

            if _dim == 1{
                return in_dt.t().to_owned();
            }else{
                return in_dt;
            }
        }else{
            panic!("Not a valid dim for 2D ndarry.");
        }
    }
}

pub struct Sigmoid<const M: usize> {
}

impl Sigmoid<1> {
    #[allow(dead_code)]
    pub fn sigmoid(mut _in: Array<f32, Dim<[usize; 1]>>) -> Array<f32, Dim<[usize; 1]>>{
        _in.iter_mut().for_each(|x| {
            if *x > 0.0 {
                *x = 1.0 / (1.0 + (-*x).exp())
            }else {
                let exp_x = (*x).exp();
                *x = exp_x / ((1.0 + exp_x))
            }
        });
        return _in;
    }
}

impl Sigmoid<2> {
    #[allow(dead_code)]
    pub fn sigmoid(mut _in: Array<f32, Dim<[usize; 2]>>) -> Array<f32, Dim<[usize; 2]>>{
        _in.iter_mut().for_each(|x| {
            if *x > 0.0 {
                *x = 1.0 / (1.0 + (-*x).exp())
            }else {
                let exp_x = (*x).exp();
                *x = exp_x / ((1.0 + exp_x))
            }
        });
        return _in;
    }
}