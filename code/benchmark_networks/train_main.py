import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import time
from functools import partial
from tqdm import tqdm
from IPython.display import clear_output 
from data_input import *
from Network_structure import *
from loss_function import *
import os

# Author: Haoming Zhang
# Here is the main part of the denoising neurl network, We can adjust all the parameter in the user-defined area.
#####################################################自定义 user-defined ########################################################

epochs = 1    # training epoch
batch_size  = 1000    # training batch size
train_num = 3000   # how many trails for train
test_num = 400     # how many trails for test
combin_num = 10    # combin EEG and noise ? times
denoise_network = 'Simple_CNN'    #   fcNN   &   Simple_CNN   &    Complex_CNN    &    RNN_lstm
result_location = r'C:/EEGdenoiseNet/code/benchmark_networks/NN_result'     #  Where to export network results
foldername = 'EMG_sCNN_10'            # the name of the target folder (should be change when we want to train a new network)

#################################################### 数据输入 Import data #####################################################

EEG_all = np.load('C:\EEG_EEGN\EEG_256hz_3400_random.npy')                              
noise_all = np.load('C:\EEG_EEGN\EOG_256hz_3400_random.npy')                              

noiseEEG_train, EEG_train, noiseEEG_test, EEG_test, test_std_VALUE = data_prepare(EEG_all, noise_all, combin_num, train_num, test_num)

################################################## optimizer adjust parameter  ####################################################
rmsp=tf.optimizers.RMSprop(lr=0.0001, rho=0.9)
adam=tf.optimizers.Adam(lr=0.00005, beta_1=0.5, beta_2=0.9, epsilon=1e-08)
sgd=tf.optimizers.SGD(lr=0.0002, momentum=0.9, decay=0.0, nesterov=False)

denoise_optimizer = adam

###################################################### train method & step ########################################################
tf.keras.backend.set_floatx('float64')

if not os.path.exists(result_location +'/'+  foldername):
  os.mkdir(result_location +'/'+  foldername)
  
@tf.function
def train_step(noiseEEG_batch,EEG_batch):

  #本次训练参数初始化  parameter initialization in one step

  mse_grads = 0
  m_loss = 0
      
 
  with tf.GradientTape() as loss_tape:
    
    M_loss =  0
    for x in range(batch_size):

      noiseeeg_batch,eeg_batch =  noiseEEG_batch[x] , EEG_batch[x]

      if denoise_network == 'fcNN':
        noiseeeg_batch = tf.reshape(noiseeeg_batch, [1,datanum])
      else:
        noiseeeg_batch = tf.reshape(noiseeeg_batch, [1,datanum,1])

      eeg_batch=tf.reshape(eeg_batch, [1,datanum,1])
      denoiseoutput = denoiseNN(noiseeeg_batch)
      denoiseoutput = tf.reshape(denoiseoutput, [1,datanum,1])                          

      m_loss = denoise_loss_mse(denoiseoutput,eeg_batch)   
      M_loss += m_loss

    
    M_loss = M_loss / float(batch_size) 
    
    # calculate gradient
    mse_grads = loss_tape.gradient(M_loss, denoiseNN.trainable_variables)
    #bp
    denoise_optimizer.apply_gradients(zip(mse_grads, denoiseNN.trainable_variables))

  return  M_loss,  mse_grads[0]  #每一条EEG的loss从此输出



def train(noiseEEG,EEG, epochs):

  # setup history variables 
  history = {}
  history['grads'], history['loss']= {}, {}
  train_mse_history, test_mse_history = [],[]
  mse_grads_history = []

  for epoch in range(epochs):
    start = time.time()

    # initialize  loss value for every epoch
    mse_grads , train_mse = 0, 0

    with tqdm(total=N_batch, position=0, leave=True) as pbar:
      
      for i in range(0,int(noiseEEG.shape[0]/batch_size)):

        #
        noiseEEG_batch,EEG_batch =  noiseEEG[batch_size*i : batch_size*(i+1)] , EEG[batch_size*i : batch_size*(i+1)]

        mse_loss_batch, mse_grads_batch = train_step(noiseEEG_batch,EEG_batch)

        # convert variables to usable format
        mse_grads_batch = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(mse_grads_batch)))).numpy()
        mse_loss_batch = tf.reduce_mean(mse_loss_batch).numpy()

        # store history 
        train_mse += mse_loss_batch/float(N_batch)
        mse_grads += mse_grads_batch/float(N_batch)

        pbar.update()
      pbar.close()


    # calculate mse loss for test set
    test_mse = 0
    for i in range (noiseEEG_test.shape[0]):
      noiseeeg_test =  noiseEEG_test[i]
      eeg_test = EEG_test[i]
    
      noiseeeg_test=noiseeeg_test.reshape(1,datanum,1)
      denoiseoutput_test = denoiseNN(noiseeeg_test)
      
      test_mse += denoise_loss_mse(eeg_test,denoiseoutput_test).numpy() 

    test_mse = test_mse/float(noiseEEG_test.shape[0])
    test_mse_history.append(test_mse) 

    # store history 
    mse_grads_history.append(mse_grads)
  

    print ('Epoch #: {}/{}, Time taken: {} secs,\n Grads: mse= {},\n Losses: train_mse= {},test_mse={}'\
             .format(epoch+1,epochs,time.time()-start , mse_grads,  train_mse, test_mse))
    

  #Generate after the final epoch
  clear_output(wait=True)

  # generate every signal in training set
  Denoiseoutput_train = []
  for i in range (noiseEEG_train.shape[0]):
    noiseeeg_train =  noiseEEG_train[i]

    noiseeeg_train = noiseeeg_train.reshape(1,datanum,1)

    denoiseoutput_train = denoiseNN(noiseeeg_train)

    denoiseoutput_train=tf.reshape(denoiseoutput_train, [ datanum])

    Denoiseoutput_train.append(denoiseoutput_train)

  if not os.path.exists(result_location +'/'+  foldername + '/' + "train_output"):
    os.mkdir(result_location +'/'+  foldername + '/' + "train_output"   )
  np.save(result_location +'/'+  foldername + '/' + "train_output" + '/' + "noiseinput_train.npy", noiseEEG_train)
  np.save(result_location +'/'+  foldername + '/' + "train_output" + '/' +  "Denoiseoutput_train.npy", Denoiseoutput_train)               #######################   地址要改！！！！！！！！
  np.save(result_location +'/'+  foldername + '/' + "train_output" + '/' +  "EEG_train.npy", EEG_train)


  # generate every signal in test set
  Denoiseoutput_test = []
  for i in range (noiseEEG_test.shape[0]):
    noiseeeg_test =  noiseEEG_test[i]
    
    noiseeeg_test=noiseeeg_test.reshape(1,datanum,1)
    denoiseoutput_test = denoiseNN(noiseeeg_test)

    denoiseoutput_test=tf.reshape(denoiseoutput_test, [ datanum])

    Denoiseoutput_test.append(denoiseoutput_test)                            
    
  if not os.path.exists(result_location +'/'+  foldername + '/' + "test_output"):
    os.mkdir(result_location +'/'+  foldername + '/' + "test_output")    
  np.save(result_location +'/'+  foldername + '/' + "test_output" +'/' + "noiseinput_test.npy", noiseEEG_test)
  np.save(result_location +'/'+  foldername + '/' + "test_output" +'/' +  "Denoiseoutput_test.npy", Denoiseoutput_test)                      #######################   地址要改！！！！！！！！
  np.save(result_location +'/'+  foldername + '/' + "test_output" +'/' + "EEG_test.npy", EEG_test)
  

  #plot train and test loss
  plt.figure()
  plt.plot(train_mse_history, 'r')
  plt.plot(test_mse_history,'b')
  plt.title('Loss history')
  plt.xlabel('Epochs')
  plt.ylabel('Loss_mse')
  plt.legend(['Train_loss', 'Test_loss'])
  plt.show()

  # Save network structure
  path = os.path.join(result_location, foldername, "denoise_model")
  tf.keras.models.save_model(denoiseNN, path)

  history['grads']['mse'] = mse_grads_history
  history['loss']['train_mse'], history['loss']['test_mse']  = train_mse_history, test_mse_history
    
  return history    


############################################################# Import network #############################################################

datanum=int(EEG_all.shape[1])
N_batch=int(EEG_train.shape[0] / batch_size)
  
if denoise_network == 'fcNN':
  denoiseNN = fcNN(datanum)

elif denoise_network == 'Simple_CNN':
  denoiseNN = simple_CNN(datanum)

elif denoise_network == 'Complex_CNN':
  denoiseNN = Complex_CNN(datanum)

elif denoise_network == 'RNN_lstm':
  denoiseNN = RNN_lstm(datanum)

else: 
  print('NN name arror')

# We have reserved an example of importing an existing network
'''
path = os.path.join(result_location, foldername, "denoised_model")
denoiseNN = tf.keras.models.load_model(path)
'''

############################################################# Running #############################################################

training_history = train(noiseEEG_train,EEG_train, epochs)
np.save(result_location + '/' + foldername + '/' + "loss_history.npy", training_history)                            