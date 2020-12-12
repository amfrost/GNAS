# sample test modified from https://medium.com/vicuesoft-techblog/how-to-use-nvidia-gpu-in-docker-to-run-tensorflow-9cf5ee279319

import sys
import numpy as np
import tensorflow as tf
from datetime import datetime

device_name = sys.argv[1]  # Choose device from cmd line. Options: gpu or cpu
shape = (int(sys.argv[2]), int(sys.argv[2]))
if device_name == "gpu":
    device_name = "/gpu:0"
else:
    device_name = "/cpu:0"
with tf.device(device_name):
    startTime = datetime.now()
    random_matrix = tf.random.uniform(shape=shape, minval=0, maxval=1)
    dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
    sum_operation = tf.reduce_sum(dot_operation)
    print('\n\n')
    print("Shape:", shape, "Device:", device_name)
    print("Time taken:", str(datetime.now() - startTime))
    