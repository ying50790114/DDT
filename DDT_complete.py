import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

import numpy as np
import cv2


def DDT(DDT_feature):

    def Calculating_descriptor_covariance(z_code=tf.zeros((4, 4, 512)), mean_vector=tf.zeros((512, 1))):
        z_code = tf.constant(z_code)
        mean_vector = tf.constant(mean_vector)
        covariance_matrixs = []
        for i in range(z_code.get_shape()[0]):
            for j in range(z_code.get_shape()[1]):
                vector = z_code[i, j, :]
                vector = tf.expand_dims(vector, axis=-1)
                temp_vector = vector - mean_vector
                covariance_matrix = tf.matmul(temp_vector, temp_vector, transpose_b=True)
                covariance_matrixs.append(covariance_matrix.numpy())
        covariance_matrixs = np.array(covariance_matrixs)
        return covariance_matrixs

    def Calculating_eigenvalues_and_eigenvector(covariance_matrixs=tf.zeros((512, 512))):
        _, eigenvector = tf.linalg.eigh(covariance_matrixs)
        return tf.expand_dims(eigenvector[0], axis=-1)

    def cal_mean_vector(z_code=tf.zeros((10, 4, 4, 512))):
        z_code = tf.constant(z_code)
        sum_vector = tf.reduce_sum(z_code, axis=[0, 1, 2])
        N, w, h, _ = z_code.shape
        vector = sum_vector / (w * h * N)
        vector = tf.expand_dims(vector, axis=-1)
        return vector.numpy()

    def test_stage(z_code=tf.zeros((4, 4, 512)), eigenvector=tf.zeros((512, 1)), mean_vector=tf.zeros((512, 1))):
        z_code = tf.constant(z_code)
        mean_vector = tf.constant(mean_vector)
        eigenvector = tf.constant(eigenvector)
        heatmap = np.ones((z_code.shape[0], z_code.shape[1]))
        for i in range(z_code.get_shape()[0]):
            for j in range(z_code.get_shape()[1]):
                vector = z_code[i, j, :]
                vector = tf.expand_dims(vector, axis=-1)
                temp_vector = vector - mean_vector
                p_value = tf.matmul(eigenvector, temp_vector, transpose_a=True)
                heatmap[i, j] = p_value.numpy()
        maxp = heatmap.max()
        minp = heatmap.min()
        heatmap = (heatmap - minp) / (maxp - minp)
        return cv2.resize(heatmap, (128, 128))

    vector = cal_mean_vector(DDT_feature)
    vector = np.array(vector)

    covariance_matrix = [Calculating_descriptor_covariance(DDT_feature[idx], vector) for idx in
                         range(DDT_feature.shape[0])]
    covariance_matrix = np.array(covariance_matrix)

    covariance_matrix = tf.constant(covariance_matrix)
    covariance_matrix = tf.reduce_mean(covariance_matrix, axis=(0, 1))
    eigenvector = Calculating_eigenvalues_and_eigenvector(covariance_matrix)

    heatmaps = []
    for idx in range(DDT_feature.get_shape()[0]):
        heatmap = test_stage(z_code=DDT_feature[idx], eigenvector=eigenvector, mean_vector=vector)
        heatmaps.append(heatmap)

    return heatmaps