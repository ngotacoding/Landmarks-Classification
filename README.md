# LANDMARKS CLASSIFICATION
This is the third project in AWS Machine Learning Fundamentals Nanodegree by Udacity.  

## Overview
In this project, I built a classification pipeline utilizing Convolutional Neural Networks (CNNs) to predict the most likely locations of user-supplied images based on any landmarks depicted in the images. The objective was to address the challenge of determining the location of photos that lack location metadata. 
I built a CNN from scratch for landmark classification, then used transfer learning for efficiency. Finally, I made a Voila app for classifying new uploaded landmark images. Utilizing the trained model, it categorizes images into one of the 50 possible landmark classes.
Training a CNN from scratch achieves a test accuracy of 51%. Transfer Learning improves the test accuracy to 77%. 

## Dataset
The dataset comprises images capturing landmarks but may also include mundane scenes like animals or plants present at those locations. It poses a challenging task for classification due to diverse image content.

## Project Structure
The project follows the following structure:

```bash
- cnn_from_scratch.ipynb   # Notebook for building CNN from scratch for landmark classification
- transfer_learning.ipynb  # Notebook for implementing transfer learning for landmark classification
- app.ipynb                # Notebook for developing the voila app integrating the trained model
- src/                     # Directory containing source files
- README.md                # Project overview and instructions

```

## Setting up Development Environment

To start developing your project, follow these steps:

Create a new Conda environment with Python 3.7.6:
```bash
conda create --name landmarks_classification -y python=3.7.6
conda activate landmarks_classification
```
Install the project requirements:
```bash
pip install -r requirements.txt
```

Install JupyterLab:
```bash
pip install jupyterlab
jupyter lab
```
