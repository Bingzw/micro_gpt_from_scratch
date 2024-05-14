# Micro GPT from Scratch

## Introduction

This project is an implementation of a simplified version of the GPT (Generative Pretrained Transformer) model from 
scratch in PyTorch. It can be used to generate text based on a given context. The goal of this project is for 
self-learning purpose and to understand the inner workings of the GPT model. 

## Setup

To set up the project, follow these steps:

1. Clone the repository
2. create a virtual environment
```
python3 -m venv venv
```
3. Install the dependencies
```
pip install -r requirements.txt
```
4. Run the project
```
python train.py
```
## Code Structure
The project consists of several Python scripts:  
- train.py: This is the main script that trains the GPT model.
- gpt.py: This script contains the implementation of the GPT model and its components, including the TransformerBlock and FeedForward classes.
- utils.py: This script contains utility functions for the project, such as functions for text generation, loss calculation, and text-to-token and token-to-text conversion.
- mha.py: This script contains the implementation of the MultiHeadAttention class, which is used in the TransformerBlock class.

## Limitations
Discuss any limitations of your project. This could be areas where your project does not perform well or features that it currently lacks.  
Future Plan
Discuss your plans for the future of the project. This could include features you plan to add or improvements you plan to make.  

## Reference
https://github.com/rasbt/LLMs-from-scratch
