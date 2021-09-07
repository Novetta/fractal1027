### Author: Alan Reiner
October, 2021

-----

This repo & notebook accompanies the Novetta/AFS blog post published [here]()

-----

Notebook:

1. Background:  CuPy, Complex Numbers, Fractals
2. CPU Implementation of Julia Set Generation
3. GPU Implmenetation using CuPy
4. Real-time Fractal Manipulation!
5. Using CuPy FFTs to find good fractals


### NOTE: This notebook should work in Colab without installing any packages _except_ for the interactive visuals. Those will only work in Jupyter or JupyterLab.

-----

## 1(a). CuPy

This hackathon submission is about using CuPy to generate fractals.  CuPy is to Numpy what cuDF is to Pandas and cuML is to scikit-learn:

* CPU: **Numpy** -> GPU: **CuPy**
* CPU: **Pandas**  ->  GPU: **CuDF**
* CPU: **Sklearn**  ->  GPU: **CuML**

CuPy is particularly useful for lower-level operations, such as linear algebra, matrix multiplications, etc.  The speedups are exactly what you would expect based on `cuDF` and `cuML` speedups -- with the right kind of data you can get multiple orders of magnitude speedup (and some less-impressive improvements for other data workflows):

![](https://miro.medium.com/max/700/1*-v7rQQd4QTAM8QV-yMhJSw.png)

Source:  https://medium.com/rapids-ai/single-gpu-cupy-speedups-ea99cbbb0cbb)

CuPy code looks exactly like Numpy.  Here's a sample:

```
import cupy as cp

# This is identical to numpy, just use `cp` instead `np`
gpu_array = cp.random.normal(size=(7, 22))
avgs = cp.mean(gpu_array, axis=0))
vars = cp.mean(cp.square(gpu_array - avgs), axis=0)
gpu_norm_array = (gpu_array - avgs) / (vars + 1e-8)

# Load directly into RAPIDS/GPU dataframe
gpu_df = cudf.DataFrame(gpu_norm_array)

# Or pull it off the GPU to use in regular numpy/pandas/sklearn;  `arr.get()` or `cp.asnumpy(arr)`
cpu_df = pd.DataFrame(gpu_norm_array.get())
```

There are even convenience methods to make your code agnostic to the underlying arrays so that you can write code that works (efficiently) regardless of whether `np` or `cp` arrays were provided.


## CuPy Has a Not-So-Secret Superpower!

The GPU-speed-with-Numpy-API alone is worth getting to know CuPy.  However, there's an exceptionally powerful feature of it:

* **You can write, compile and run C++/CUDA code on-the-fly**. CuPy takes care of all of it without having to ever run a C++/CUDA compiler yourself.

Sometimes what you're trying to do with large arrays is not easily done by simple compositions of numpy/cupy operations.  Or perhaps those operations don't have the flexibility you need.  Or perhaps you just know how to optimize the CUDA code itself.  There are two levels to this capability:

1. Define an element-wise operation in C++, letting CuPy handle block & grid logistics for you
2. Define a "raw kernel" which is writing all the logic yourself, requiring your own grid/block logic, but also allowing you to use shared memory, etc.

For this application, we only need an elementwise kernel, and we'll see how easy it is!



## 1(b). Fractal Math (and Complex Numbers)

"Julia Sets" are fractals produced by treating each pixel of an image as a point in the complex plane, and running it through a loop 1,000 times (square-and-add-constant) to determine how long it takes to "escape" towards infinity.  The set of points that oscillate near the origin before escaping form incredible patterns.

This process is highly-parallelizable, since each pixel's escape time can be computed independently of all other pixels, and consists of only simple algebra in a loop.  This makes it a prime candidate for GPU acceleration.

Therefore, we represent the grid of points in the complex plane as a two, two-dimensional CuPy arrays (real & imaginary), and define a `cp.Elementwise(...)` operation to push the computation into the GPU where each CUDA core computes each pixel in parallel.  It might seem at first that you could perform all these operations with standard array compositions, but it should be observed that the innermost loop will run a different number of times for each pixel.   This is why we use an `cp.Elementwise(...)` kernel.


### Background: Complex Numbers

All complex numbers, `a + bi` can be represented in a polar form:

* r e<sup>iθ</sup>

Variable `r` is the absolute value ("modulus") of the vector, and `θ` is the "argument" (angle).  Because of the magic of the imaginary exponent, multiplying two complex numbers together is the same as multiplying the absolute values and *adding* the angles.

The image below show this in the complex plane (x-axis is real, y-axis is imaginary).

![](https://community.topcoder.com/i/education/091806_03.gif)



### Fractal generation - Julia Sets

In this exercise, we pick a constant (complex) `c`, and treat every pixel as a point on the complex plane (`z`).  Then apply the following function over and over again:

* f(z) -> z<sup>2</sup> + c

Based on the diagram above, we know that each iteration doubles the angle of `z`, squares its absolute value, and then translates it by `c`.  The absolute value of some points will simply just grow ("escape") towards infinity.  But some points, the translation by `c` could offset some of the changes of the z<sup>2</sup> operation, keeping its motion "oscillating" near the origin ... at least for a little while.

The fractal is simply a map of how long each point in the complex plane takes to "escape" for the given value of `c`.  Thes set of points that oscillate instead of escaping is called a **Julia Set**:

![](https://www.researchgate.net/profile/Christian-Bauckhage/publication/272679245/figure/fig1/AS:294937234034690@1447329924538/Visualization-of-a-Julia-set.png)

The most amazing part of these fractals is the infinite resolution of its structures.  Infinitely small perturbations of `z`-values can push a point from stable to escaping. 

This gif from wikipedia is for the related "Mandelbrot Set," showing how it maintains its structure no matter how far you zoom in, to the point that in some locations, a window of size 10<sup>-100</sup>-by-10<sup>-100</sup> contains its own mandelbrot set:

![](https://upload.wikimedia.org/wikipedia/commons/a/a4/Mandelbrot_sequence_new.gif)

(Source: Wikipedia on Mandelbrot Sets)
