import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import time
from functools import partial
from tqdm import tqdm
from IPython.display import clear_output 
from data_prepare import *
from Network_structure import *
from loss_function import *
import os  
import math
import datetime

# Author: Haoming Zhang
# Here is the part of denoiseNet training process

@tf.function
def train_step(model, noiseEEG_batch, EEG_batch, optimizer , denoise_network, batch_size, datanum):

    #本次训练参数初始化  parameter initialization in one step

    mse_grads = 0
    m_loss = 0
      
 
    with tf.GradientTape() as loss_tape:
    
        M_loss =  0
        for x in range(noiseEEG_batch.shape[0]):
    
            noiseeeg_batch,eeg_batch =  noiseEEG_batch[x] , EEG_batch[x]

            if denoise_network == 'fcNN':
                noiseeeg_batch = tf.reshape(noiseeeg_batch, [1,datanum])
            else:
                noiseeeg_batch = tf.reshape(noiseeeg_batch, [1,datanum,1])

            eeg_batch=tf.reshape(eeg_batch, [1,datanum,1])
            denoiseoutput = model(noiseeeg_batch)
            denoiseoutput = tf.reshape(denoiseoutput, [1,datanum,1])                          

            m_loss = denoise_loss_mse(denoiseoutput,eeg_batch)   
            M_loss += m_loss

    
        M_loss = M_loss / float(noiseEEG_batch.shape[0]) 
        
        # calculate gradient
        mse_grads = loss_tape.gradient(M_loss, model.trainable_variables)
        #bp
        optimizer.apply_gradients(zip(mse_grads, model.trainable_variables))

    return  M_loss,  mse_grads[0]  #每一条EEG的loss从此输出

def test_step(model, noiseEEG_test, EEG_test):

  denoiseoutput_test = model(noiseEEG_test)
  loss = denoise_loss_mse(EEG_test, denoiseoutput_test)
  #loss_rrmset = denoise_loss_rrmset(denoiseoutput_test, EEG_test)

  return denoiseoutput_test, loss#, loss_rrmset


def train(model, noiseEEG,EEG, noiseEEG_val, EEG_val, epochs, batch_size,optimizer, denoise_network, result_location, foldername, train_num):

    # setup history variables and save history in a npy film
    history = {}
    history['grads'], history['loss']= {}, {}
    train_mse_history, val_mse_history = [],[]
    mse_grads_history = []
    val_mse_min = 100.0      # any number bigger than 1

    # save history to tensorboard
    # current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = result_location +'/'+foldername +'/'+ train_num + '/train'
    val_log_dir = result_location +'/'+foldername +'/'+ train_num + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    batch_num = math.ceil(noiseEEG.shape[0]/batch_size)
    
    datanum = noiseEEG.shape[1]
    for epoch in range(epochs):
        start = time.time()

        # initialize  loss value for every epoch
        mse_grads , train_mse = 0, 0

        with tqdm(total=batch_num, position=0, leave=True) as pbar:
    
            for n_batch in range(batch_num):

                #
                if n_batch == batch_num:
                    noiseEEG_batch,EEG_batch =  noiseEEG[batch_size*n_batch :] , EEG[batch_size*n_batch :]
                else:
                    noiseEEG_batch,EEG_batch =  noiseEEG[batch_size*n_batch : batch_size*(n_batch+1)] , EEG[batch_size*n_batch : batch_size*(n_batch+1)]

                mse_loss_batch, mse_grads_batch = train_step(model, noiseEEG_batch,EEG_batch, optimizer, denoise_network, batch_size , datanum)

                # convert variables to usable format
                mse_grads_batch = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(mse_grads_batch)))).numpy()
                mse_loss_batch = tf.reduce_mean(mse_loss_batch).numpy()

                # store history 
                train_mse += mse_loss_batch/float(batch_num)
                mse_grads += mse_grads_batch/float(batch_num)

                pbar.update()
            pbar.close()

        # store train history 
        mse_grads_history.append(mse_grads)
        train_mse_history.append(train_mse)

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_mse, step=epoch)


        # calculate mse loss for validation set
        #denoiseoutput, val_mse, loss_rrmset = test_step(model, noiseEEG_val, EEG_val)
        denoiseoutput, val_mse = test_step(model, noiseEEG_val, EEG_val)

        #store validation history
        val_mse_history.append(val_mse) 

        with val_summary_writer.as_default():   # record validation loss to tensorboard
            tf.summary.scalar('loss', val_mse, step=epoch)

        if epoch>epochs*0.8 and float(val_mse) < val_mse_min:  # if epoch_number > 0.8*all_epoch_number begin to save the best model  ## for SCNN or CCNN in EMG we should save the first or second model. 
            print('yes,smaller ', float(val_mse) ,val_mse_min)
            val_mse_min = float(val_mse)
            saved_model = model

            path = os.path.join(result_location, foldername, train_num, "denoise_model")
            tf.keras.models.save_model(model, path)
            print('Best model has been saved')

        print ('Epoch #: {}/{}, Time taken: {} secs,\n Grads: mse= {},\n Losses: train_mse= {}, val_mse={}'\
                     .format(epoch+1,epochs,time.time()-start , mse_grads,  train_mse, val_mse))

            
    #Generate after the final epoch
    clear_output(wait=True)

    #save history to dict
    history['grads']['mse'] = mse_grads_history
    history['loss']['train_mse'], history['loss']['val_mse']  = train_mse_history, val_mse_history
        
    return saved_model, history    