# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 12:13:35 2023

@author: sanat
"""

import numpy as np
import pandas as pd
import torch
from typing import List, Tuple
import random
import math
import os
import torch.nn.functional as F
import pathlib

# Loading and preprocessing
dataset_file = pathlib.Path("dinos.txt") 

with open(dataset_file, 'r') as f:
    dinos = f.readlines()

vocab_size = 27
n_a = 10

keys = list("abcdefghijklmnopqrstuvwxyz_")
itoc = {k: v for k,v in enumerate(keys)}


def character_encoding(txt):
    """
    Parameters:
    ---------------
    txt: character
        The character that needs to be encoded
    Return an encoded numpy.array of given character
    """
    keys = list("abcdefghijklmnopqrstuvwxyz_")
    values = ""
    values = [x for x in np.arange(vocab_size)]

    # making a dictionary
    encodings = {}
    for key, value in zip(keys, values):
        encodings[key] = value

    # print(encodings)

    char = torch.zeros((vocab_size, 1))
    char[encodings[txt]] = 1
    return char


def word_encoding(txt):
    """
    Return word encoding of given word
    """

    # The size of the output will be (27, 2) if word is of 2 character
    out = torch.zeros((vocab_size, len(txt)))

    for i in range(len(txt)):
        out[:, i] = character_encoding(txt[i]).reshape(-1,)

    return out.T


def dataset_encoding(txt):

    # here txt is list of words
    # replacing \n
    txt = [x.replace("\n", "_").lower() for x in txt]

    # the output will be a list of transformed words
    out = []
    for word in txt:
        out.append(word_encoding(word))
    return torch.vstack(out)


encodings = dataset_encoding(dinos)


# ------------------------TRAINING-------------------------
# ------------------------ONE ITERATION--------------------
n_a = 30
n_emb = 27
a_0 = torch.zeros((1, n_a))  # 1, 30
Wax = torch.zeros((n_emb, n_a))  # 27, 30
Waa = torch.zeros((n_a, n_a))  # 30, 30
bx = torch.rand((1, n_a)) * 0.01
x_1 = encodings[0].view(1, -1)
Way = torch.rand((n_a, n_emb)) * 0.01
by = torch.rand((1, n_emb)) * 0.01

a_1 = a_0 @ Waa + x_1 @ Wax + bx  # 1, 30
a_2 = a_1 @ Waa + x_1 @ Wax + bx


y_1 = a_1 @ Way + by
y_pred = torch.softmax(y_1, dim=1)


# ------------------------TRAINING LOOP---------------------

# rnn with x += y, information from previous loss also :)
# abc -> abc_ and so will be prediction
dinos = [x.replace("\n", "_").lower() for x in dinos]


# -------------PARAMETERS-----------------------
# torch.manual_seed(6789)
n_a = 30
n_emb = 27
a_0 = torch.zeros((1, n_a))  # 1, 30
x_0 = torch.zeros((1, n_emb))
Wax = torch.randn((n_emb, n_a)) * 0.01  # 27, 30
Waa = torch.randn((n_a, n_a)) * 0.01  # 30, 30
bx = torch.randn((1, n_a)) * 0.01
# x_1 = encodings[0].view(1, -1)
Way = torch.randn((n_a, n_emb)) * 0.01
by = torch.randn((1, n_emb)) * 0.01

# parameters = [Wax, Waa, bx, Way, by]
parameters = [Way, by, Waa, Wax, bx]
for p in parameters:
    p.requires_grad = True


def train(epochs):
    losses = []
    for epoch in range(epochs):

        predictions = []
        labels = []
        # forward pass
        for idx, word in enumerate(dinos[:3]):
            word = word_encoding(word)
            # print(word.shape)
            prediction = torch.zeros(word.shape[0], n_emb)

            characters = word.shape[0]
            a = a_0
            # x = x_0

            x_new = torch.zeros_like(word)
            x_new[1:] = word[:-1]
            precalculated = x_new @ Wax
            for i in range(characters):
                a = a @ Waa + precalculated[i] + bx
                y = a @ Way + by
                prediction[i] = y

            labels.append(word)
            # print(prediction.shape)
            # print("-------------")
            predictions.append(prediction)

        predictions = torch.vstack(predictions)
        labels = torch.vstack(labels)

        # calculating loss
        loss = F.cross_entropy(predictions, labels)
        losses.append(loss.item())
        if epoch % 100 == 0:
            print(f"loss -> {loss.item()}")

        # update the parameters
        for p in parameters:
            p.grad = None

        loss.backward(retain_graph=True)

        lr = 0.01 if epoch < 500 else 0.1

        for p in parameters:
            p.data -= lr * p.grad
    return losses


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


@torch.no_grad()
def sample(no_of_examples=1):
    sampled_words = []
    for i in range(no_of_examples):
        words = [] # list of ssamples words
        a = a_0
        x = x_0
        
        while True:
            a = a @ Waa + x @ Wax + bx
            y = a @ Way + by
            x = y
            y = torch.softmax(y, dim=1)
            word = torch.multinomial(y, num_samples=1).item()  # int value
            char = itoc[word]
            words.append(char)
            
            if char == '_':
                break
        print(f"{''.join(words)}\n")
        sampled_words.append(''.join(words))
    return sampled_words
            

print("\nsampling from untrained model\n")
print("----------------------------------------")
print("----------------------------------------")
untrained_sampled_words = sample(5)

losses = train(1000)
import matplotlib.pyplot as plt
plt.plot(range(len(losses)), losses)

print("----------------------------------------")
print("----------------------------------------")
print("\nsampling from trained model\n")
trained_sampled_words = sample(10)


# ---------------------------DEMONSTRATION--------------------
predictions = []
labels = []
# torch.autograd.set_detect_anomaly(True)
for i in range(0):
    for idx, word in enumerate(dinos[:1]):
        word = word_encoding(word)
        # print(word.shape)
        prediction = torch.zeros(word.shape[0], n_emb)

        characters = word.shape[0]
        a = a_0
        x = x_0

        x_new = torch.zeros_like(word)
        x_new[1:] = word[:-1]
        precalculated = x_new @ Wax
        # print(precalculated.shape)
        for i in range(characters):
            # a = a @ Waa + x @ Wax + bx
            # y = a @ Way + by
            # x = word[i].view(1, -1) if i != characters-1 else 0
            # # print(f'the normal x is -> {x=}')
            # # x += y
            # # print(f'the updated x is -> {x=}')
            # prediction[i] = y

            a = a @ Waa + precalculated[i].view(1, 30) + bx
            a = a @ Waa + precalculated[i] + bx
            y = a @ Way + by
            prediction[i] = y

        loss = F.cross_entropy(prediction, word)
        print(f"the loss is -> {loss.item()}")
        print("hiiiiii")
        for p in parameters:
            p.grad = None
            
        print("45678965789056789")
        loss.backward()
        print("-================")
        for p in parameters:
            p.data -= 0.01 * p.grad

        labels.append(word)
        print(prediction.shape)
        print("-------------")
        predictions.append(prediction)


# predictions = torch.vstack(predictions)
# labels = torch.vstack(labels)

# print(predictions.shape, labels.shape)
# -------------------------------------------------------------------
