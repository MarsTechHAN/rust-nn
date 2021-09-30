pub trait Layer<T> {
    fn forward(&mut self, _tensor: T) -> T;
    fn backward(&mut self, _grad: T) -> T;
}