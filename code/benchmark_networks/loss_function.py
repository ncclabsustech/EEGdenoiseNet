import tensorflow as tf



################################# loss functions ##########################################################

def denoise_loss_mse(denoise, clean):      
  loss = tf.losses.mean_squared_error(denoise, clean)
  return tf.reduce_mean(loss)

