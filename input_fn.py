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
#   repeats forever
#   small values of shuffle buffer results better number of combinations
#   new whales in the sense unidentified whales are removed from train set and will added later
    ds1 = tf.data.TFRecordDataset(train_file)
    ds1 = ds1.map(parse_fn)
#    ds1 = ds1.filter(lambda x,y:tf.equal(y,"new_whale"))
    ds1 = ds1.repeat()
    ds1 = ds1.shuffle(10)
    itr1 = ds1.make_one_shot_iterator()
    img1,l1 = itr1.get_next()

    ds2 = tf.data.TFRecordDataset(train_file)
    ds2 = ds2.map(parse_fn)
#    ds2 = ds2.filter(lambda x,y:tf.equal(y,"new_whale"))
    ds2 = ds2.repeat()
    ds2 = ds2.shuffle(10)
    itr2 = ds2.make_one_shot_iterator()
    img2,l2 = itr2.get_next()

    y = tf.cast(tf.equal(l1,l2),tf.int32)
   
    im_set = (img1,img2)
    f, l=tf.contrib.training.stratified_sample([im_set], y,
                            batch_size=20, enqueue_many=False, queue_capacity=1000,
                                            target_probs=tf.convert_to_tensor([0.5,0.5]),  threads_per_queue=4)

    return f, l

def pred_ip_fn():
    dummy_lables= [1 for i in range(7960)]
    ds1 = tf.data.Dataset.from_tensor_slices((testfiles, dummy_lables))
    ds1 = ds1.map(imgprcs, 4)
    ds1 = ds1.repeat(15697)
    ts_itr = ds1.make_one_shot_iterator()

    ds2 = tf.data.Dataset.from_tensor_slices((trainfiles, labels))
    ds2 = ds2.map(imgprcs,4)
    ds2 = ds2.repeat(7690)
    tr_itr = ds2.make_one_shot_iterator()

    def jumble(xy1, xy2):
        (x1,y1) = xy1
        (x2,y2) = xy2
        return (x1,x2),(y1,y2)
    ds3 = tf.data.Dataset.zip((ds1,ds2))
    ds3 = ds3.map(jumble)
    itr = ds3.make_one_shot_iterator()

    imgs, lbls = itr.get_next()
    #print(imgs)
    return imgs, None 
     
