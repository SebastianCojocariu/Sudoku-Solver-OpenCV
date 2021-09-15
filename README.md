## REQUIREMENTS
```bash
* opencv-python==4.5.1.48
* numpy==1.16.6
* onnx==1.9.0
* onnx-tf==1.8.0
* tf2onnx==1.7.1
* matplotlib=3.2.2
* sortedcollections==1.2.1
* mpmath==1.1.0
* scipy=1.5.0
* argparse=1.1
* skimage=0.16.2
* more-itertools==8.4.0
```

## HOW TO RUN THE CODE

### TASK1

Python script: task_1.py

Function of interest: task1(image_path, save_path=None, verbose=0)
```bash
--image_path: path to an image we want to apply the algorithm. Default is None
--save_path: path where to save the output. Default is None
--path_dir: path from where to extract the files to run the algorithm on. Default is ./dataset/images/classic/
--save_dir: path where to save the output for the files from path_dir. Default is ./dataset/predictions/classic/
--no_file: the number of file from path_dir where we want to apply our algorithm. It saves the output on save_dir (it must be specified). Default is None
--verbose: a flag to show different data along the running of the algorithm. Default is 0
```

* To run all the tests, just run: python3 task_1.py (This will get all the 20 images from ./dataset/images/classic/ and save the outputs on ./dataset/predictions/classic/)
* To run a specific image (specified by path), just run: python3 task_1.py --image_path <path_to_the_image> <optional/save_path for the output>
* To run a specific image (specified by number from path_dir), just run: python3 task_1.py --no_file <the number of file> --path_dir <path to the directory where to find the file>

### TASK2
Python script: task_2.py
Function of interest: task2(image_path, save_path=None, verbose=0)

```bash
--image_path: path to an image we want to apply the algorithm. Default is None
--save_path: path where to save the output. Default is None
--path_dir: path from where to extract the files to run the algorithm on. Default is ./dataset/images/jigsaw/
--save_dir: path where to save the output for the files from path_dir. Default is ./dataset/predictions/jigsaw/
--no_file: the number of file from path_dir where we want to apply our algorithm. It saves the output on save_dir (it must be specified). Default is None
--verbose: a flag to show different data along the running of the algorithm. Default is 0
```

* To run all the tests, just run: python3 task_2.py (This will get all the 15 images from ./dataset/images/jigsaw/ and save the outputs on ./dataset/predictions/jigsaw/)
* To run a specific image (specified by path), just run: python3 task_2.py --image_path <path_to_the_image> <optional/save_path for the output>
* To run a specific image (specified by number from path_dir), just run: python3 task_2.py --no_file <the number of file> --path_dir <path to the directory where to find the file>

### TASK3
Python script: task_3.py
Function of interest: task3(image_path, save_path_text, save_path_image, template_given_path, template_own_path, verbose=0):

```bash
--image_path: path to an image we want to apply the algorithm. Default is None.
--template_own_path: the path to my own template (this will be used with a template matching algorithm to get a transformation matrix on the new template). Default is ./own_template.jpg.
--template_given_path: the path to the given template. Default is ./train/cube/template.jpg.
--save_path_tex: the path where to save the text output. Default is None.
--save_path_image: the path where to save the image output. Default is None.
--path_dir: path from where to extract the files to run the algorithm on. Default is ./dataset/images/cube/
--save_dir: path where to save the output for the files from path_dir. Default is ./dataset/predictions/cube/
--no_file: the number of file from path_dir where we want to apply our algorithm. It saves the output on save_dir (it must be specified). Default is None
--verbose: a flag to show different data along the running of the algorithm. Default is 0
```

* To run all the tests, just run: python3 task_3.py (This will get all the 5 images from ./dataset/images/cube/ and save the outputs on ./dataset_predictions/cube/)
* To run a specific image (specified by path), just run: python3 task_3.py --image_path <path_to_the_image> <optional:save_path_text and save_path_image for the output>
* To run a specific image (specified by number from path_dir), just run: python3 task_3.py --no_file <the number of file> --path_dir <path to the directory where to find the file>

To use a given template, use --template_given_path argument to specify its path.

CAUTION: the NN model must be located at "./mnist_model.onnx" (or be specified it s location in the task3 function otherwise)