use mnist::*;
use ndarray::prelude::*;
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use rand::prelude::*;
use ndarray_stats::QuantileExt;
use image::*;
use show_image::{make_window_full, Event, WindowOptions};

fn main() {
    
    let (input, output, test_input, test_output) = mnist_as_ndarray();
    println!("Successfully unpacked the MNIST dataset into Array2<f32> format!");

    // Lets see an example of the parsed MNIST dataset on the training and testing data
    let mut rng = rand::thread_rng();
    let mut num: usize= rng.gen_range(0,input.nrows());
    println!("Input record #{} has a label of {}",num,output.slice(s![num,..]));
    display_img(input.slice(s![num,..]).to_owned().into_shape((28,28)).unwrap());
    
    let mut dnn = DeepNeuralNetwork::default(input, output, vec![784,128,64,10]);    
    dnn.train();
    let test_result = dnn.evaluate(test_input);
    compare_results(test_result, test_output);

}

struct DeepNeuralNetwork {
    input: Array2<f32>,
    output: Array2<f32>, 
    sizes: Vec<usize>,
    iterations: usize,
    learnrate: f32,
    w: Vec<Array2<f32>>,
    z: Vec<Array2<f32>>,
    a: Vec<Array2<f32>>,
}

impl DeepNeuralNetwork {

    fn default(input: Array2<f32>, output: Array2<f32>, sizes: Vec<usize>) -> DeepNeuralNetwork {
        let mut dnn = DeepNeuralNetwork {
            input: input,
            output: output,
            sizes: sizes,
            iterations: 750,
            learnrate: 0.01,
            w: vec![],
            z: vec![Array::zeros((1,1)); 3],
            a: vec![Array::zeros((1,1)); 4],
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

fn display_img(input: Array2<f32>) {
    let output_img = bw_ndarray2_to_image(input);
    let window_options = WindowOptions {
        name: "image".to_string(),
        size: [300, 300],
        resizable: true,
        preserve_aspect_ratio: true,
    };
    println!("\nPlease hit [ ESC ] to quit window:");
    let window = make_window_full(window_options).unwrap();
    window.set_image(output_img, "test_result").unwrap();

    for event in window.events() {
        if let Event::KeyboardEvent(event) = event {
            if event.key == show_image::KeyCode::Escape {
                break;
            }
        }
    }

    show_image::stop().unwrap();
}


/// Helper function for transition from an normalized NdArray3<f32> structure to an `Image::RgbImage`
fn bw_ndarray2_to_image(arr: Array2<f32>) -> RgbImage {
    assert!(arr.is_standard_layout());

    println!("{:?}",arr.dim());
    let (height, width) = arr.dim();
    println!("producing an image of size: ({},{})",width, height);
    let mut img: RgbImage = ImageBuffer::new(width as u32, height as u32);
    for y in 0..height {
        for x in 0..width {
            let val = (arr[[y,x]] * 255.) as u8;
            img.put_pixel(x as u32, y as u32, image::Rgb([val, val, val]));
        }
    }
    img
}


