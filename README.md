# tflite-tools
TFLite model analyser &amp; memory optimizer.

The tool is able to produce a short analysis of a Tensorflow Lite (v3) models, which includes:
* Information about intermediate tensors that need to be present in RAM (excludes weights, as they can be read directly
 from the model file.)
* Operator evaluation schedule (as given by the operator order in the model file), along with tensors that need to present at every step of execution and the amount of 
memory occupied by them.
* Plot memory usage during evaluation, detailing sizes of input and output tensors for each operator, as well as other
 tensors that are present in memory (see example image at the end of 'Example output' section).

The analysis can be printed to the standard output or to a set of CSV files using the `--csv` option.

Additionally, the tool can:
* Modify the model to minimise peak memory usage by reordering operators in the model file (`--optimize` option).
* Simulate code-book quantization by clustering the weights into `n` centroids, and replacing each weight with the 
closest centroid value. Note that this is done for each weight matrix separately and biases are left untouched.

The tool also offers an API through the `TFLiteModel` class --- see `def main()` in `tflite_tools.py` for example 
usage.

## Setup
The tool requires Python 3.6+ and a few dependencies, as described in `Pipfile`.
To create a new virtual environment with correct dependencies, run the following the root of the repository:

```
pipenv install
```

(requires `pipenv`, which you can install through your system's package manager or via `pip`: `pip install pipenv`)

## Usage
```
% pipenv shell
% python tflite_tools.py --help
usage: tflite_tools.py [-h] [-i INPUT_PATH] [-o OUTPUT_PATH]
                       [--clusters CLUSTERS] [--optimize]

TFLite model analyser & memory optimizer

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_PATH         input model file (.tflite)
  -o OUTPUT_PATH        output model file (.tflite)
  --clusters CLUSTERS   cluster weights into n-many values (simulate code-book
                        quantization)
  --optimize            optimize peak working set size
  --csv CSV_OUTPUT_FOLDER
                        output model analysis in CSV format into the specified
                        folder
  --plot PLOT_FILE      plot memory usage for each operator during the
                        execution
```

## Example output
```
% python tflite_tools.py -i quantized_model.tflite -o quantized_model_optimized.tflite

Tensor information (weights excluded):
+----+-----------------------+-----------------+-----------------+
| Id |         Tensor        |      Shape      | Size in RAM (B) |
+----+-----------------------+-----------------+-----------------+
|  1 |       Conv1/Relu      | (1, 30, 30, 16) |          14,400 |
|  2 |      Conv1_input      |  (1, 32, 32, 3) |           3,072 |
|  4 |       Conv2/Relu      | (1, 28, 28, 16) |          12,544 |
|  5 |      FC2/BiasAdd      |     (1, 10)     |              10 |
|  7 |      FC2/Softmax      |     (1, 10)     |              10 |
|  8 |    activation/Relu    |     (1, 128)    |             128 |
|  9 | max_pooling2d/MaxPool | (1, 14, 14, 16) |           3,136 |
+----+-----------------------+-----------------+-----------------+

Operator execution schedule:
+------------------------+-------------------------+----------------+
| Operator (output name) | Tensors in memory (IDs) | Memory use (B) |
+------------------------+-------------------------+----------------+
|       Conv1/Relu       |          [1, 2]         |         17,472 |
|       Conv2/Relu       |          [1, 4]         |         26,944 |
| max_pooling2d/MaxPool  |          [4, 9]         |         15,680 |
|    activation/Relu     |          [8, 9]         |          3,264 |
|      FC2/BiasAdd       |          [8, 5]         |            138 |
|      FC2/Softmax       |          [5, 7]         |             20 |
+------------------------+-------------------------+----------------+
Current peak memory usage: 26,944 B

```

```
% python tflite_tools.py -i example_model.tflite --plot example_working_set.png
```

![Example working set plot](https://github.com/oxmlsys/tflite-tools/raw/master/example_working_set.png)
