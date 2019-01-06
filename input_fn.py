import pandas as pd
import numpy as np
import tensorflow as tf
import glob
import itertools as it
from operator import itemgetter
import threading
import time
from siamese_model import SiameseNet, model_fn

tf.logging.set_verbosity(tf.logging.INFO)

train_file= "./train_data/whaleDetection_train_data.tfrecords"
test_file= "./test_data/whaleDetection_test_data.tfrecords"
tr_itr=tf.python_io.tf_record_iterator(train_file)
ts_itr=tf.python_io.tf_record_iterator(test_file)

def parse_fn(record):
    kys_to_fts = {"image_name":tf.FixedLenFeature((), tf.string),
                    "label":tf.FixedLenFeature((), tf.string, default_value=""),
                            "image_raw":tf.FixedLenFeature((), tf.string)}
    parsed = tf.parse_single_example(record, kys_to_fts)

    lbl=tf.cast(parsed["label"], tf.string)
    img=tf.io.decode_raw(parsed["image_raw"], tf.uint8)
    img=tf.cast(tf.reshape(img,(64,64,3)),tf.float32)
    return img, lbl

def ip_fn():
#   this repetition of code is required to cover the same images of same class    
    ds1 = tf.data.TFRecordDataset(train_file)
    ds1 = ds1.map(parse_fn)
    ds1 = ds1.filter(lambda x,y:tf.equal(y,"new_whale"))
    ds1 = ds1.repeat(100)
    ds1 = ds1.shuffle(1000)
    itr1 = ds1.make_one_shot_iterator()
    img1,l1 = itr1.get_next()

    ds2 = tf.data.TFRecordDataset(train_file)
    ds2 = ds2.map(parse_fn)
    ds2 = ds2.filter(lambda x,y:tf.equal(y,"new_whale"))
    ds2 = ds2.repeat(100)
    ds2 = ds2.shuffle(1000)
    itr2 = ds2.make_one_shot_iterator()
    img2,l2 = itr2.get_next()

    y = tf.cast(tf.equal(l1,l2),tf.int32)
   
    im_set = (img1,img2)
    f, l=tf.contrib.training.stratified_sample([im_set], y,
                            batch_size=20, enqueue_many=False, queue_capacity=1000,
                                            target_probs=tf.convert_to_tensor([0.5,0.5]),  threads_per_queue=4)

    return f, l

def pred_ip_fn():
#   this repetition of code is required to cover the same images of same class    
    ds1 = tf.data.TFRecordDataset(test_file)
    ds1 = ds1.map(parse_fn)
    ts_itr = ds1.make_one_shot_iterator()
    img1,l1 = ts_itr.get_next()

    ds2 = tf.data.TFRecordDataset(train_file)
    ds2 = ds2.map(parse_fn)
    ds2 = ds2.filter(lambda x,y:tf.equal(y,"new_whale"))
    tr_itr = ds2.make_one_shot_iterator()
    img2,l2 = tr_itr.get_next()
   
    for img1, l1 in ts_itr:
        for img2, l2 in tr_itr:
            return (img1,img2), None
     
