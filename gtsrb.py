N_CLASSES= 43
RESIZED_IMAGE = (32,32)

import matplotlib.pyplot as plt
import glob
from skimage.color import rgb2lab
from skimage.transform import resize
from collections import namedtuple
import numpy as np
np.random.seed(101)


Dataset = namedtuple('Dataset' , ['X', 'y'])

def to_tf_format(imags) :
    return np.stack([img[:,:,np.newaxis] for img in imags], axis=0).astype(np.float32)

def read_dataset_ppm(rootpath, n_labels, resize_to) :
    images = []
    labels = []

    for c in range(n_labels) :
        full_path = rootpath + '/' + format(c, '05d') + '/'

        for img_name in glob.glob(full_path + '*.ppm') :
            img = plt.imread(img_name).astype(np.float32)
            img = rgb2lab(img/255.0)[:,:,0]
            if resize_to :
                img = resize(img, resize_to, mode ='reflect')
            label = np.zeros((n_labels,) , dtype=np.float32)
            label[c] = 1.0

            images.append(img.astype(np.float32))
            labels.append(label)
    return Dataset(X= to_tf_format(images).astype(np.float32),
                    y= np.matrix(labels).astype(np.float32))

dataset = read_dataset_ppm('./GTSRB/Final_Training/Images', N_CLASSES, RESIZED_IMAGE)

print(dataset.X.shape)
print(dataset.y.shape)

plt.imshow(dataset.X[0,:,:,:].reshape(RESIZED_IMAGE))
print(dataset.y[0,:])

from sklearn.model_selection import train_test_split

idx_train, idx_test = train_test_split(range(dataset.X.shape[0]), test_size = 0.25, random_state = 101)
X_train = dataset.X[idx_train, :, :, :]
X_test = dataset.X[idx_test, :, : , :]
y_train = dataset.y[idx_train,:]
y_test = dataset.y[idx_test, :]

print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)

def minibatcher(X, y, batch_size, shuffle) :
    assert X.shape[0] == y.shape[0]
    n_sample = X.shape[0]

    if shuffle:
        idx = np.random.permutation(n_sample)
    else :
        idx = list(range(n_sample))

    for k in range(int(np.ceil(n_sample/batch_size))) :
        from_idx = k * batch_size
        to_idx = (k+1)*batch_size
        yield X[idx[from_idx:to_idx], : , :, :], y[idx[from_idx:to_idx], :]

for mb in minibatcher(X_train, y_train, 100, True) :
    print(mb[0].shape, mb[1].shape)


import tensorflow as tf

def fc_no_activation_layer(in_tensors, n_units) :
    w = tf.get_variable('fc_W',
        [in_tensors.get_shape()[1], n_units],
        tf.float32,
        tf.contrib.layers.xavier_initializer())
    b = tf.get_variable('fc_B',
        [n_units,],
        tf.float32,
        tf.constant_initializer(0.0))
    return tf.matmul(in_tensors,w) + b

def fc_layer(in_tensors, n_units) :
    return tf.nn.leaky_relu(fc_no_activation_layer(in_tensors, n_units))

def conv_layer(in_tensors, kernel_size, n_units) :
    w = tf.get_variable('conv_W',
        [kernel_size, kernel_size, in_tensors.get_shape()[3], n_units],
        tf.float32,
        tf.contrib.layers.xavier_initializer())
    b = tf.get_variable('conv_B',
        [n_units,],
        tf.float32,
        tf.constant_initializer(0.0))
    return tf.nn.leaky_relu(tf.nn.conv2d(in_tensors, w, [1,1,1,1], 'SAME')+b)

def maxpool_layer(in_tensors, sampling) :
    return tf.nn.max_pool(in_tensors, [1,sampling, sampling, 1], [1,sampling, sampling,1] , 'SAME')


def dropout(in_tensors, keep_proba, is_training) :
    return tf.cond(is_training, lambda:tf.nn.dropout(in_tensors, keep_proba), lambda:in_tensors)


def model(in_tensors, is_training) :
    with tf.variable_scope('l1') :
        l1 = maxpool_layer(conv_layer(in_tensors,5,32), 2)
        l1_out = dropout(l1, 0.8, is_training)

    with tf.variable_scope('l2') :
        l2 = maxpool_layer(conv_layer(in_tensors,5,64), 2)
        l2_out = dropout(l2, 0.8, is_training)

    with tf.variable_scope('flatten') :
        l2_out_flat = tf.layers.flatten(l2_out)

    with tf.variable_scope('l3') :
        l3 = fc_layer(l2_out_flat, 1024)
        l3_out = dropout(l3, 0.6, is_training)

    with tf.variable_scope('out') :
        out_tensors = fc_no_activation_layer(l3_out, N_CLASSES)

    return out_tensors


from sklearn.metrics import classification_report, confusion_matrix

def train_model(X_train, y_train, X_test, y_test, learning_rate, max_epochs, batch_size) :
    in_X_tensors_batch = tf.placeholder(tf.float32, shape = (None, RESIZED_IMAGE[0], RESIZED_IMAGE[1], 1))
    in_y_tensors_batch = tf.placeholder(tf.float32, shape = (None, N_CLASSES))
    is_training = tf.placeholder(tf.bool)

    logits = model(in_X_tensors_batch, is_training)
    out_y_pred = tf.nn.softmax(logits)
    loss_score = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=in_y_tensors_batch)
    loss = tf.reduce_mean(loss_score)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.Session() as session :
        session.run(tf.global_variables_initializer())

        for epoch in range(max_epochs) :
            print("Epoch", epoch)
            tf_score = []

            for mb in minibatcher(X_train,y_train, batch_size, shuffle=True) :
                tf_output = session.run([optimizer, loss],
                                        feed_dict={in_X_tensors_batch : mb[0], in_y_tensors_batch:mb[1], is_training: True})
                tf_score.append(tf_output[1])
            print(" train loss_score" , np.mean(tf_score))


        print("TEST SET PERFORMANCE")
        y_test_pred, test_loss = session.run([out_y_pred, loss],
                                            feed_dict={in_X_tensors_batch: X_test, in_y_tensors_batch:y_test, is_training: False})
        print("test_loss_score" , test_loss)

        y_test_pred_classified = np.argmax(y_test_pred, axis=1).astype(np.int32)
        y_test_true_classified = np.argmax(y_test, axis=1).astype(np.int32)
        print(classification_report(y_test_true_classified, y_test_pred_classified))

train_model(X_train, y_train, X_test, y_test, 0.001, 5, 100)