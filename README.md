# docker_MNIST
This is a repository  with docker backend in python that uses tensorflow and the MNIST data to recognize handwritten digits in a separate frontend layer using gradio.

## In simiulation picture:
![Frontend that recognize handwritten digits. ](https://github.com/eliascarrasco1227/docker_MNIST/blob/main/execution%20pictures%20and%20videos/frontend_picture.png)

![Executing a multi-container application with Docker. ](https://github.com/eliascarrasco1227/docker_MNIST/blob/main/execution%20pictures%20and%20videos/Executing%20a%20multi-container%20application%20with%20Docker.png)


## Introduction 
This project contains a deep learning inference system within Docker containers.
Using a pre-trained neural network model for digit recognition, this system comprises two main containers:
one for inference using a Flask API and another for a Gradio-based web
interface. The containers communicate through a Docker Compose network setup. The main goal of the project is to illustrate the encapsulation
of deep learning models and graphical interfaces within isolated Docker
environments to facilitate modular deployment. Key configuration files,
Dockerfiles, and performance-related observations are discussed in detail.