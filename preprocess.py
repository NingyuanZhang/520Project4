import os
from PIL import Image
import random
import numpy as np
from scipy import misc
import matplotlib.image as mpimg
from time import time
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

raw_data_dir = 'BikaQiu_raw/'
resized_data_dir = 'resized_data/'

train_data_dir = 'resized_data/train_set/'
test_data_dir = 'resized_data/test_set'

data_dir = 'data/'

_width = 64
_height = 64
_d = 3
_number_of_train_images = 0
_number_of_test_image = 0

for file_name in os.listdir(train_data_dir):
    if file_name.endswith('.jpg'):
        _number_of_train_images = _number_of_train_images + 1
print('Number of training samples', _number_of_train_images)

for file_name in os.listdir(test_data_dir):
    if file_name.endswith('.jpg'):
        _number_of_test_image = _number_of_test_image + 1
print('Number of test samples', _number_of_test_image, '\n')

# number of clusters
n_colors = 16


def resize_data(raw_data_dir, resized_data_dir):
    for file_name in os.listdir(raw_data_dir):
        i = random.random()
        file_path = os.path.join(raw_data_dir, file_name)
        resized_data_path = os.path.join(resized_data_dir, file_name)
        image = Image.open(file_path)
        image = np.array(image)
        hight, width, d = np.shape(image)
        pad_size = abs(hight - width) // 2
        if hight > width:
            pad_dims = ((0, 0), (pad_size, pad_size))
        else:
            pad_dims = ((pad_size, pad_size), (0, 0))

        channel_one = image[:, :, 0]
        channel_two = image[:, :, 1]
        channel_three = image[:, :, 2]

        channel_one = np.pad(channel_one, pad_dims, 'constant', constant_values=(255,255))
        channel_two = np.pad(channel_two, pad_dims, 'constant', constant_values=(255,255))
        channel_three = np.pad(channel_three, pad_dims, 'constant', constant_values=(255,255))

        image = np.dstack((channel_one, channel_two, channel_three))

        resized_image = misc.imresize(image, (64, 64))

        misc.imsave(resized_data_path, resized_image)


def save_data_as_numpy(train_data_dir, test_data_dir):
    # get and save original training  data
    train_data_original = []
    for file_name in os.listdir(train_data_dir):
        if file_name.endswith('.jpg'):
            file_path = os.path.join(train_data_dir, file_name)
            image = mpimg.imread(file_path)
            image_array = np.reshape(image, (_width*_height, _d))
            train_data_original.append(image_array)
    train_data_original = np.array(train_data_original, dtype=np.float64) / 255

    np.save('data\\train_data_original', train_data_original)
    print('saved training original data!')
    print('shape of training original data:', np.shape(train_data_original), '\n')

    # get and save original test data
    test_data_original = []
    for file_name in os.listdir(test_data_dir):
        if file_name.endswith('.jpg'):
            file_path = os.path.join(test_data_dir, file_name)
            image = mpimg.imread(file_path)
            image_array = np.reshape(image, (_width*_height, _d))
            test_data_original.append(image_array)
    test_data_original = np.array(test_data_original, dtype=np.float64) / 255

    np.save('data\\test_data_original', test_data_original)
    print('saved test original data!')
    print('shape of Test original data:', np.shape(test_data_original), '\n')


    def rgb2gray(rgb):
        return np.dot(rgb[..., :3],[0.299, 0.587, 0.114])

    # To get training gray image set
    train_data_gray = []
    # number, length, d = train_data_original.shape
    for i in range(_number_of_train_images):
        gray_image = rgb2gray(train_data_original[i])
        train_data_gray.append(gray_image)

    np.save('data\\train_data_gray', train_data_gray)
    print('saved training gray data!')
    print('shape of training gray data:', np.shape(train_data_gray), '\n')

    # To get test gray image set
    test_data_gray = []
    # number, length, d = train_data_original.shape
    for i in range(_number_of_test_image):
        gray_image = rgb2gray(test_data_original[i])
        test_data_gray.append(gray_image)

    np.save('data\\test_data_gray', test_data_gray)
    print('saved test gray data!')
    print('shape of test gray data:', np.shape(test_data_gray), '\n')

    # use K-means to cluster original image
    train_data_original_for_kmeans = np.reshape(train_data_original, (_number_of_train_images * _width * _height, _d))
    train_data_color_sample = shuffle(train_data_original_for_kmeans, random_state=0)[:_number_of_train_images*_width*_height]

    t0 = time()
    kmeans = KMeans(n_clusters= n_colors, random_state=0).fit(train_data_color_sample)
    print("K-means done in %0.3fs." % (time() - t0))
    print('size of cluster centers :', np.shape(kmeans.cluster_centers_), '\n')

    # label training set
    t0 = time()
    train_data_labels = kmeans.predict(train_data_original_for_kmeans)
    train_data_labels = np.reshape(train_data_labels, (_number_of_train_images, _width * _height))
    print("Labels Training set done in %0.3fs." % (time() - t0))
    print('size of labels of training set', np.shape(train_data_labels), '\n')

    # label test set
    test_data_original_for_kmeans = np.reshape(test_data_original, (_number_of_test_image * _width * _height, _d))
    t0 = time()
    test_data_labels = kmeans.predict(test_data_original_for_kmeans)
    test_data_labels = np.reshape(test_data_labels, (_number_of_test_image, _width * _height))
    print("Labels Test set done in %0.3fs." % (time() - t0))
    print('size of labels of test set', np.shape(test_data_labels), '\n')

    # get clustered training image
    print('Computing clustered training image.')
    train_data_clustered = np.zeros((_number_of_train_images, _width * _height, _d))
    for i in range(_number_of_train_images):
        for j in range(_width * _height):
            train_data_clustered[i][j] = kmeans.cluster_centers_[train_data_labels[i][j]]
    print('size of train clustered set: ', np.shape(train_data_clustered), '\n')

    # get clustered test image
    print('Computing clustered test image.')
    test_data_clustered = np.zeros((_number_of_test_image, _width * _height, _d))
    for i in range(_number_of_test_image):
        for j in range(_width * _height):
            test_data_clustered[i][j] = kmeans.cluster_centers_[test_data_labels[i][j]]
    print('size of test clustered set: ',np.shape(test_data_clustered), '\n')

    # save train clustered data
    np.save('data\\train_data_clustered', train_data_clustered)
    print('saved train clustered set.')
    print('shape of training clustered image:', np.shape(train_data_clustered), '\n')

    # save test clustered data
    np.save('data\\test_data_clustered', test_data_clustered)
    print('saved test clustered set.')
    print('shape of test clustered image:', np.shape(test_data_clustered), '\n')

    # save train data labels
    np.save('data\\train_data_labels', train_data_labels)
    print('saved train labels.')
    print('shape of training data labels :', np.shape(train_data_labels), '\n')

    # save test data labels
    np.save('data\\test_data_labels', test_data_labels)
    print('saved test labels.')
    print('shape of training data labels :', np.shape(test_data_labels), '\n')

    # save label dictionary
    label_dictionary = kmeans.cluster_centers_
    np.save('data\\label_dictionary', label_dictionary)
    print('saved label dictionary.')
    print('shape of label dictionary :', np.shape(label_dictionary), '\n')


def show_image(data_dir):
    label_dictionary = np.load(data_dir + 'label_dictionary.npy')
    print('size of label dictionary: ',np.shape(label_dictionary),'\n')

    train_data_gray = np.load(data_dir + 'train_data_gray.npy')
    print('size of training  gray: ',np.shape(train_data_gray),'\n')

    train_data_original = np.load(data_dir + 'train_data_original.npy')
    test_data_original = np.load(data_dir + 'test_data_original.npy')

    train_data_labels = np.load(data_dir + 'train_data_labels.npy')
    print('size of training data labels: ',np.shape(train_data_labels),'\n')

    train_data_clustered = np.load(data_dir + 'train_data_clustered.npy')
    test_data_clustered = np.load(data_dir + 'test_data_clustered.npy')

    test_data_gray = np.load(data_dir + 'test_data_gray.npy')
    print('size of test data gray: ',np.shape(test_data_gray),'\n')

    test_data_labels = np.load(data_dir + 'test_data_labels.npy')

    x = 0

    test = test_data_clustered[x]
    plt.imshow(np.reshape(test,(_width,_height,_d)))
    plt.axis('off')
    plt.title('test cluster')
    plt.show()

    test = test_data_original[x]
    plt.imshow(np.reshape(test,(_width,_height,_d)))
    plt.axis('off')
    plt.title('test original')
    plt.show()

    test = test_data_gray[x]
    plt.imshow(np.reshape(test,(_width,_height)),cmap='Greys_r')
    plt.axis('off')
    plt.title('test gray')
    plt.show()

    test = train_data_clustered[x]
    plt.imshow(np.reshape(test,(_width,_height,_d)))
    plt.title('train cluster')
    plt.axis('off')

    plt.show()
    test = train_data_original[x]
    plt.imshow(np.reshape(test,(_width,_height,_d)))
    plt.axis('off')
    plt.title('original')
    plt.show()

    test = train_data_gray[x]
    plt.imshow(np.reshape(test,(_width,_height)),cmap='Greys_r')
    plt.axis('off')
    plt.title('train gray')
    plt.show()



if __name__ == '__main__':
    # resize_data(raw_data_dir, resized_data_dir)
    # save_data_as_numpy(train_data_dir,test_data_dir)
    show_image(data_dir)
