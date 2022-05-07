# Introduction to Image Processing

## Poznan University of Technology, Institute of Robotics and Machine Intelligence

<p align="center">
  <img width="180" height="180" src="./readme_files/logo.png">
</p>

# **Final project: fruit counting**

## TASK

The project task is to prepare an algorithm for detecting and counting the fruits present in the images. To simplify the task there are only 3 types of fruits in the data set:
- apples
- bananas
- oranges

All images were captured "from above" but from different heights. In addition, the images differ in light levels and, of course, the amount of fruit.

An example image from the dataset and the correct detection result for it is shown below:


<p align="center">
  <img width="400" height="300" src="https://user-images.githubusercontent.com/65020389/167257585-a8076246-2ed2-4800-8a8f-26bea49f476a.jpg">
</p>



```bash
PS C:\Projects\WdPO> python -u "c:\Projects\WdPO\test.py"
Apple:  2 Banana:  1 Orange:  1
```


### Libraries

- [OpenCV](https://docs.opencv.org/master/) version 4.5.3.56
- [NumPy](https://numpy.org/) version 1.19.5
- [Click](https://palletsprojects.com/p/click/) version 7.1.2
- [tgdm](https://tqdm.github.io/) version 4.62.3
