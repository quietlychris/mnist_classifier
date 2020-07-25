### MNIST Classifier

A basic classifier for the MNIST data set, which is included in the `data/` folder.

The classifier achieves around ~94% accuracy in appx. 8.4s on the `--release` build over 3000 iterations, which seems faster and somewhat more consistent than the code supplied in the tuturial. My system monitor shows this code to be primarily single-threaded, so there's likely some performance gains to be found there. The code's also not exactly ergonomic to use as a library, so I think there are some improvements to be made there. 

There's also an option to check the MNIST parsing by creating a window with a random record's image and printing the associated one-hot encoded vector. The `minifb` library had some common dependencies that may need to be installed, please see [here](https://github.com/emoon/rust_minifb) for instructions. 

This started as a clone of the "Neural Network from Scratch with NumPy and MNIST" tutorial [here](https://mlfromscratch.com/neural-network-tutorial/#/), but got a bit off track. Notably, the Rust `ndarray` library doesn't accept the same kind of arguments as the NumPy `np.dot()` function, so the backpropagation section looks somewhat different. 

Code was tested on Pop!_OS 20.04, please file an issue if something's not working on your system. 