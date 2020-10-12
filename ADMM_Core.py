# -*- coding: utf-8 -*-

import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from unet2 import UNet     
import cv2 


def layer_x(v_hat,u):
    x_wav = np.asarray(v_hat) - np.asarray(u)
    print('the shape of x_wav is {}'.format(x_wav.shape))
    model = UNet().create_model(x_wav.shape)
    x_hat = model.output
    
    return x_hat
    
def layer_v(x_hat,u):
    v_wav = np.asarray(x_hat) + np.asarray(u)
    model = UNet().create_model(v_wav.shape)
    v_hat = model.output
    return v_hat


def layer_u(u,x_hat,v_hat):
    u = u + np.asarray(x_hat) -np.asarray(v_hat)
    return u
        
def ADMMNet(inputs):
    
    u=np.zeros((3,256,256)).astype(np.float32)
    norm=u[:,:,0]+u[:,:,1]+u[:,:,2]
    norm=np.linalg.norm(norm)
    x_hat =np.asarray(inputs).astype(np.float32)
    v_hat = x_hat

    x_hat = layer_x(v_hat,u)
    v_hat = layer_v(x_hat,u)
    u = layer_u(u,x_hat,v_hat)
    '''
    for i in range(10):
        if norm>1e-3:
            x_hat = layer_x(v_hat,u)
            v_hat = layer_v(x_hat,u)
            u = layer_u(u,x_hat,v_hat)
       
        else:
            print('The training is stopped')
    
    if norm > 1e-3:
        print('It is a bad experiment')
    '''
    return x_hat,norm 


def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.
     
     From W x H x C [0...255] to C x W x H [0..255]
    '''
    ar = np.array(img_PIL)
 
    if len(ar.shape) == 3:
        ar = ar.transpose(2,0,1)
    else:
        
        ar = ar[None, ...]
    return ar.astype(np.float32)

def train(learning_rate=0.001):
    
    print('Loading the Image')
    img=Image.open('data/training/109.jpg')
    cropped = img.crop((256,0,512,256))
    x_train=pil_to_np(cropped)
    
    print('Starting training')
    out,norm = ADMMNet(x_train)
    with tf.name_scope("optimize"):
        # 优化器
        optim = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(norm)
    with tf.Session() as s:
        s.run(tf.global_variables_initializer())
        s.run(optim)
    plt.imshow(out)
    
    
if __name__ == '__main__':
    train()
    
    
    
