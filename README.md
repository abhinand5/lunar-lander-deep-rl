# Lunar Lander with Deep Reinforcement Learning

Lunar Lander is a very interesting environment in OpenAI Gym. The main objective is to make an AI spacecraft (Agent) learn by itself to land smoothly in the right place in the simulated environment provided by OpenAI Gym.

> Algorithm Used: Deep Q Learning

## Getting Started
The reproduce the results in your local machine, follow the steps.

### Prerequisites
1. Python 3.7
2. pip
3. venv

### Built with
* Python 3
* OpenAI Gym
* Tensorflow 2
* Numpy


### Installing 
#### Option 1: (Linux/MacOS)
First create a virtual env. You can name it anything.

1. `virtualenv --system-site-packages -p python3 ./venv`

2. `source ./venv/bin/activate`

Install listed requirements from the `requirements.txt` file.

3. `pip install -r requirements.txt`

(Windows users just have to change dir structure accordingly, or you can use **WSL**)


#### Option 2: (Manual install)
If you already have a virtual env set up for deep learning and you just want to install some extra stuff. (Or if Option 1 didn't work)

> Manually try to install all much needed dependecies yourself from the respective official web pages.

## Usage

After installing all the required packages and frameworks, you're ready to use the code.

To Train,
1. `cd src`
2. `python agent.py`

To open up tensorboard logs, go to your terminal and type..

`tensorboard --logdir logs/`

Then open your browser and go to,
`http://localhost:6006/`

To make inference of a trained model,
1. `cd src`
2. `python inference.py`

Now the result gifs will be stored in the results folder.

## Authors
* **Abhinand Balachandran**