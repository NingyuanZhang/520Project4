import tensorflow as tf
from sklearn.utils import shuffle
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt

isTrain = False
isTrain = True

data_dir = 'data/'

_width = 64
_height = 64
_d = 3
_number_of_train_images = 0
_number_of_test_image = 0

label_dictionary = np.load(data_dir+'label_dictionary.npy')
print('size of label dictionary: ', np.shape(label_dictionary), '\n')

train_data_gray = np.load(data_dir+'train_data_gray.npy')
print('size of training  gray: ', np.shape(train_data_gray), '\n')

train_data_labels = np.load(data_dir+'train_data_labels.npy')
print('size of training data labels: ', np.shape(train_data_labels), '\n')


test_data_gray = np.load(data_dir+'test_data_gray.npy')
print('size of test data gray: ', np.shape(test_data_gray), '\n')

test_data_labels = np.load(data_dir+'test_data_labels.npy')
print('size of test_data_labels: ', np.shape(test_data_labels), '\n')

tf.reset_default_graph()

# 64x64x1
with tf.name_scope('Input'):
    x = tf.placeholder(tf.float32, [None, _width * _height], name='x-input')
    y = tf.placeholder(tf.float32, [None, _width * _height], name='y-input')
    with tf.name_scope('keep_prob'):
        keep_prob = tf.placeholder(tf.float32, name='keep-prob')
    with tf.name_scope('learning_rate'):
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

x_image = tf.reshape(x, shape=[-1, _width, _height, 1])

# # 64x64x1 -> 64x64x64
with tf.name_scope('Conv1'):
    with tf.name_scope('w_conv1'):
        w_conv1 = tf.Variable(tf.truncated_normal([3,3,1,32], stddev=0.1), name='w_con1')
    with tf.name_scope('b_conv1'):
        b_conv1 = tf.Variable(tf.zeros([32])+0.1, name='b_conv1')
    with tf.name_scope('h_conv1'):
        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x_image, w_conv1, strides=[1, 1, 1, 1], padding='SAME'), b_conv1))


# # # 32x32x64 -> 32x32x128    64 64 128 -> 32 32 128
# with tf.name_scope('Conv2'):
#     with tf.name_scope('w_conv2'):
#         w_conv2 = tf.Variable(tf.truncated_normal([3,3,32,64],stddev=0.1),name='w_con1')
#     with tf.name_scope('b_conv2'):
#         b_conv2 = tf.Variable(tf.zeros([64]) + 0.1,name='b_conv1')
#     with tf.name_scope('h_conv1'):
#         conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_conv2, strides=[1,2,2,1], padding='SAME'), b_conv2))
#
# # 32 32 256
# with tf.name_scope('Conv3'):
#     with tf.name_scope('w_conv3'):
#         w_conv3 = tf.Variable(tf.truncated_normal([3,3, 64, 16], stddev=0.1),name='w_con1')
#     with tf.name_scope('b_conv3'):
#         b_conv3 = tf.Variable(tf.zeros([16]) + 0.1,name='b_conv1')
#     with tf.name_scope('h_conv1'):
#         conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_conv3, strides=[1,1,1,1], padding='SAME'), b_conv3))

conv3 = tf.reshape(conv1, shape=[-1, 64*64*32])
#
# # 32x32x128 -> 16x16x128
# with tf.name_scope('max_pool_1'):
#     conv2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#     conv2_flat = tf.reshape(conv2,shape=[-1, 16*16*128])

with tf.name_scope('FC1'):
    with tf.name_scope('W_FC1'):
        w_fc1 = tf.Variable(tf.truncated_normal([64*64*32, _width * _height]), name='W_FC1')
    with tf.name_scope('B_FC1'):
        b_fc1 = tf.Variable(tf.zeros([_width * _height])+0.01, name='B_FC1')
    with tf.name_scope('h_FC1'):
        fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(conv3, w_fc1), b_fc1))
    with tf.name_scope('Dropout'):
        fc1 = tf.nn.dropout(fc1, keep_prob)

with tf.name_scope('FC2'):
    with tf.name_scope('W_FC2'):
        w_fc2 = tf.Variable(tf.truncated_normal([_width * _height, _width * _height]), name='W_FC2')
    with tf.name_scope('B_FC2'):
        b_fc2 = tf.Variable(tf.zeros([_width * _height])+0.01, name='B_FC2')
    with tf.name_scope('h_FC2'):
        output = tf.nn.bias_add(tf.matmul(fc1, w_fc2), b_fc2)
        output = tf.nn.sigmoid(output)


with tf.name_scope('loss'):
    # loss = tf.reduce_sum(tf.square(y - output))
    loss = tf.reduce_mean(tf.abs(y/32 - output))
    tf.summary.scalar('loss', loss)

with tf.name_scope('Train'):
    optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss)

merged = tf.summary.merge_all()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    if isTrain:
        print('Training...')
        train_writer = tf.summary.FileWriter('logs/train', sess.graph)
        test_writer = tf.summary.FileWriter('logs/test', sess.graph)

        batch_size = 4
        n_batch = len(train_data_gray) // batch_size
        # test_data_gray, test_data_labels = shuffle(test_data_gray, test_data_labels, random_state=0)

        for epoch in range(80000):
            train_data_gray, train_data_labels = shuffle(train_data_gray, train_data_labels, random_state=0)

            for i in range(n_batch):
                batch_xs = train_data_gray[i*batch_size : (i+1)*batch_size]
                batch_ys = train_data_labels[i*batch_size : (i+1)*batch_size]
                sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0, learning_rate: 0.0003})

                summary_train = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
                train_writer.add_summary(summary_train, int(epoch * n_batch + i + 1))

                summary_test = sess.run(merged, feed_dict={x: test_data_gray, y: test_data_labels, keep_prob: 1.0})
                train_writer.add_summary(summary_test, int(epoch * n_batch + i + 1))

            epoch_loss_train = sess.run(loss, feed_dict={x: train_data_gray, y: train_data_labels, keep_prob: 1.0})
            epoch_loss_test = sess.run(loss,feed_dict={x: test_data_gray,y: test_data_labels,keep_prob: 1.0})
            print('Epoch: ' + str(epoch) + ' train_loss: ' + str(epoch_loss_train) + ' test_loss: ' + str(epoch_loss_test))

        save_path = saver.save(sess, 'ModelSaver/model.ckpt')

    else:
        print("Inference...")
        saver.restore(sess, 'ModelSaver/model.ckpt')
        test_image = test_data_gray[0:1, :]

        plt.figure(1)
        plt.clf()
        plt.imshow(np.reshape(test_image, (_width, _height)), cmap='Greys_r')
        plt.title('Gray image')
        plt.axis('off')

        predicted_label = sess.run(output, feed_dict={x: test_image, keep_prob: 1.0})
        predicted_label = predicted_label - 1e-5

        recreated_image = np.zeros((_width*_height, _d))
        for i in range(_width*_height):
            recreated_image[i] = label_dictionary[int(32*predicted_label[0][i])]

        plt.figure(2)
        plt.clf()
        plt.imshow(np.reshape(recreated_image, (_width, _height, _d)))
        plt.title('colorized image')
        plt.axis('off')
        plt.show()

































