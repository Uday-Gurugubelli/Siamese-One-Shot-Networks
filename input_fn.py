import pandas as pd
import numpy as np
import tensorflow as tf
import glob
import itertools as it
from operator import itemgetter
import threading
import time
train_file= "./train_data/whaleDetection_train_data.tfrecords"
test_file= "./test_data/whaleDetection_test_data.tfrecords"

from one_shot import SiameseNet, model_fn

tr_itr=tf.python_io.tf_record_iterator(train_file)
tr_itr1=tf.python_io.tf_record_iterator(train_file)
ts_itr=tf.python_io.tf_record_iterator(test_file)
count=0
for rc in tr_itr:
    count = count+1
print("num of records:",count)
for rc in ts_itr:
    count = count+1
print("num of records:",count)
def parse_fn(rcrd):
    ex=tf.train.Example()
    ex.ParseFromString(rcrd)
    [lbl]=ex.features.feature['label'].bytes_list.value
    lbl = tf.cast(lbl, tf.string)
    [raw]=ex.features.feature['image_raw'].bytes_list.value
    img=tf.cast(tf.io.decode_raw(raw, tf.int64), tf.uint8)
    img=tf.cast(tf.reshape(img,(64,64,1)),tf.float32)
#img=tf.image.per_image_standardization(img)
    img=tf.reshape(img,[64,64])
    return img, lbl
def b_ip_fn():
    f, l=tf.contrib.training.stratified_sample(tr_product_list, tf.convert_to_tensor(tr_label_list),
            batch_size=20, enqueue_many=True, queue_capacity=100,
            target_probs=tf.convert_to_tensor([0.5,0.5]),  threads_per_queue=2)
    return f, l


est = tf.estimator.Estimator(model_fn=model_fn, model_dir="./oneshot_model_dir/")
for _ in range(1):
    est.train(input_fn = b_ip_fn, steps = 100000)
    est.evaluate(input_fn=b_ip_fn,steps=100)
