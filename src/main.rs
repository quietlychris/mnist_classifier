use mnist::*;
use ndarray::prelude::*;
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use rand::prelude::*;
use ndarray_stats::QuantileExt;
use minifb::{Key, Window, WindowOptions, ScaleMode};

fn main() {
    
    let (input, output, test_input, test_output) = mnist_as_ndarray();
    println!("Successfully unpacked the MNIST dataset into Array2<f32> format!");

    // Lets see an example of the parsed MNIST dataset on the training and testing data
    let mut rng = rand::thread_rng();
    let mut num: usize= rng.gen_range(0,input.nrows());
    println!("Input record #{} has a label of {}",num,output.slice(s![num,..]));
    display_img(input.slice(s![num,..]).to_owned());

    num = rng.gen_range(0,test_input.nrows());
    println!("Test record #{} has a label of {}",num,test_output.slice(s![num,..]));
    display_img(test_input.slice(s![num,..]).to_owned());
    
    //let mut dnn = DeepNeuralNetwork::default(input, output, vec![784,128,64,10]);    
    //dnn.train();
    //let test_result = dnn.evaluate(test_input);
    //compare_results(test_result, test_output);

}

struct DeepNeuralNetwork {
    input: Array2<f32>,
    output: Array2<f32>, 
    sizes: Vec<usize>,
    iterations: usize,
    learnrate: f32,
    w: Vec<Array2<f32>>,
    z: Vec<Array2<f32>>,
    a: Vec<Array2<f32>>
}

impl DeepNeuralNetwork {

    fn default(input: Array2<f32>, output: Array2<f32>, sizes: Vec<usize>) -> DeepNeuralNetwork {
        let mut dnn = DeepNeuralNetwork {
            input: input,
            output: output,
            sizes: sizes,
            iterations: 3000,
            learnrate: 0.01,
            w: vec![],
            z: vec![Array::zeros((1,1)); 3],
            a: vec![Array::zeros((1,1)); 4]
        };
        dnn.w.push(Array::random((784, 128),Uniform::new(-0.1, 0.1)));
        dnn.w.push(Array::random((128, 64),Uniform::new(-0.2, 0.2)));
        dnn.w.push(Array::random((64,10),Uniform::new(-0.5, 0.5)));
        dnn

    }

    fn sizes(mut self, sizes: Vec<usize>) -> Self {
        self.sizes = sizes;
        self
    }

    fn iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    fn learnrate(mut self, learnrate: f32) -> Self {
        self.learnrate = learnrate;
        self
    }

    #[inline]
    fn forward_pass(&mut self, num: usize, batch_size: usize) {
 
        self.a[0] = self.input.slice(s![num..num+batch_size,..]).to_owned();

        self.z[0] = self.a[0].dot(&self.w[0]);
        self.a[1] = self.z[0].clone().mapv(|x| sigmoid(x));

        self.z[1] = self.a[1].dot(&self.w[1]);
        self.a[2] = self.z[1].clone().mapv(|x| sigmoid(x));

        self.z[2] = self.a[2].dot(&self.w[2]);
        self.a[3] = self.z[2].clone().mapv(|x| sigmoid(x));

    }

    #[inline]
    fn backwards_pass(&mut self, num: usize, batch_size: usize, iteration: usize) {

        let error = &self.a[3] - &self.output.slice(s![num..num+batch_size,..]); // [60_000, 10]
        println!("Training iteration #{}, % error: {}",iteration, &error.sum().abs() / batch_size as f32);
        let delta2 = &error * &self.z[2].mapv(|x| sigmoid_prime(x)) * self.learnrate;
        // self.w[2] = &self.w[2] - 
        let dw2 = &self.a[2].t().dot(&delta2); 
        // println!("dw2 shape: {:?}",dw2.dim());

        let delta1 = delta2.dot(&self.w[2].t()) * self.z[1].mapv(|x| sigmoid_prime(x));
        let dw1 = &self.a[1].t().dot(&delta1);
        
        let delta0 = delta1.dot(&self.w[1].t()) * self.z[0].mapv(|x| sigmoid_prime(x));
        let dw0 = &self.a[0].t().dot(&delta0); 

        self.w[2] -= dw2;
        self.w[1] -= dw1;
        self.w[0] -= dw0;

    }
    
    fn train(&mut self) {
        let mut rng = rand::thread_rng();
        let batch_size = 200;
        for iteration in 0..self.iterations {
            let mut num: usize= rng.gen_range(0,self.input.nrows() - batch_size);
            self.forward_pass(num,batch_size);
            self.backwards_pass(num,batch_size, iteration);
        }
    }

    fn evaluate(mut self, input: Array2<f32>) -> Array2<f32> {
        self.input = input;
        self.forward_pass(0,10_000);
        self.a.last().unwrap().clone()
    }
    

}



#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn sigmoid_prime(x: f32) -> f32 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

#[inline]
pub fn softmax(array: &mut Array2<f32>) {
    for j in 0..array.nrows() {
        let sum = array.slice(s![j,..]).sum();
        for i in 0..array.ncols() {
            array[[j,i]] = array[[j,i]] / sum; 
        } 
    }
}

fn compare_results(mut actual: Array2<f32>, ideal: Array2<f32>) {
    softmax(&mut actual);
    let mut correct_number = 0;
    for i in 0..actual.nrows() {
        let result_row = actual.slice(s![i, ..]);
        let output_row = ideal.slice(s![i, ..]);

        if (result_row.argmax() == output_row.argmax())
        {
            correct_number += 1;
        }
    }
    println!(
        "Total correct values: {}/{}, or {}%",
        correct_number,
        actual.nrows(),
        (correct_number as f32) * 100. / (actual.nrows() as f32)
    );
}

fn mnist_as_ndarray() -> (Array2<f32>,Array2<f32>,Array2<f32>,Array2<f32>) {
    let (trn_size, rows, cols) = (60_000, 28, 28);
    let tst_size = 10_000;

    // Deconstruct the returned Mnist struct.
    // YOu can see the default Mnist struct at https://docs.rs/mnist/0.4.0/mnist/struct.MnistBuilder.html
    let Mnist { trn_img, trn_lbl, tst_img, tst_lbl, .. } = MnistBuilder::new()
        .label_format_one_hot()
        .finalize();

    // Convert the returned Mnist struct to Array2 format
    let trn_lbl: Array2<f32> = Array2::from_shape_vec((trn_size, 10),trn_lbl)
        .expect("Error converting labels to Array2 struct")
        .map(|x| *x as f32);   
    // println!("The first digit is a {:?}",trn_lbl.slice(s![image_num, ..]) );
    
    // Can use an Array2 or Array3 here (Array3 for visualization)
    let trn_img = Array2::from_shape_vec((trn_size, 784),trn_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.);   
    // println!("{:#.0}\n",trn_img.slice(s![image_num, .., ..]));

    // Convert the returned Mnist struct to Array2 format
    let tst_lbl: Array2<f32> = Array2::from_shape_vec((tst_size, 10),tst_lbl)
        .expect("Error converting labels to Array2 struct")
        .map(|x| *x as f32);   
    
    let tst_img = Array2::from_shape_vec((tst_size, 784),tst_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.);   

    (trn_img, trn_lbl, tst_img, tst_lbl)
}

fn display_img(input: Array1<f32>) {

    let img_vec: Vec<u8> = input.to_vec().iter().map(|x| (*x * 256.) as u8).collect();
    // println!("img_vec: {:?}",img_vec);
    let mut buffer: Vec<u32> = Vec::with_capacity(28*28);
    for px in 0..784 {
            let temp: [u8; 4] = [img_vec[px], img_vec[px], img_vec[px], 255u8];
            // println!("temp: {:?}",temp);
            buffer.push(u32::from_le_bytes(temp));
    }

    let (window_width, window_height) = (600, 600);
    let mut window = Window::new(
        "Test - ESC to exit",
        window_width,
        window_height,
        WindowOptions {
            resize: true,
            scale_mode: ScaleMode::Center,
            ..WindowOptions::default()
        },
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });

    // Limit to max ~60 fps update rate
    window.limit_update_rate(Some(std::time::Duration::from_micros(16600)));

    while window.is_open() && !window.is_key_down(Key::Escape) && !window.is_key_down(Key::Q) {
        // We unwrap here as we want this code to exit if it fails. Real applications may want to handle this in a different way
        window
            .update_with_buffer(&buffer, 28, 28)
            .unwrap();
    }
}


