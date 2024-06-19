
from tensorflow.keras.datasets import mnist
import cv2
import numpy as np

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()


null_space_5x4 = np.zeros(shape=(5*28, 4*28))


zeros_index = []
for i in range(len(y_train)):
    if y_train[i] == 0:
        zeros_index.append(i)


ones_index = []
for i in range(len(y_train)):
    if y_train[i] == 1:
        ones_index.append(i)


# label_index = []
for k in range(1000):
    t = 0
    zero_index_in_train = zeros_index[k]
    row_z = np.random.randint(5)
    col_z = np.random.randint(4)
    for i in range(5):
        for j in range(4):
            if  i == row_z and j == col_z:
                null_space_5x4[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = x_train[zero_index_in_train]
                # label_index.append([i * 28, (i + 1) * 28, j * 28, (j + 1) * 28])
            else:
                # while t in zeros_index:
                #     t = np.random.randint(60000)
                # null_space_5x4[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = x_train[t]
                # t = np.random.randint(60000)
                null_space_5x4[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = x_train[ones_index[k]]
    cv2.imwrite('/home/fagc2267/PycharmProjects/Triplet_loss/DDT_Mnist/5x4_Mnist/new/'+str(k)+'.png', null_space_5x4)
    # np.save('D:/dataset/5x4_Mnist/label_0_999',label_index)
# cv2.namedWindow("synthesis", 0)
# cv2.imshow("synthesis", null_space_5x4)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # 02x2
# for k in range(1000, 2000):
#     t = 0
#     zero_index_in_train = zeros_index[k]
#     # 選左上角
#     row_z = np.random.randint(5-1)
#     col_z = np.random.randint(4-1)
#
#     for i in range(5):
#         for j in range(4):
#             if i == row_z and j == col_z:
#                 resized = cv2.resize(x_train[zero_index_in_train], (56, 56))
#                 cv2.imshow("resized", resized)
#                 null_space_5x4[i * 28:(i + 2) * 28, j * 28:(j + 2) * 28] = resized
#                 label_index.append([i * 28, (i + 2) * 28, j * 28, (j + 2) * 28])
#             elif (i == row_z + 1 and j == col_z + 1) or (i == row_z + 1 and j == col_z) or (i == row_z and j == col_z + 1):
#                 print()
#             else:
#                 while t in zeros_index:
#                     t = np.random.randint(60000)
#                 null_space_5x4[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = x_train[t]
#                 t = np.random.randint(60000)
#     cv2.imwrite('D:/dataset/5x4_Mnist/img/'+str(k)+'.png', null_space_5x4)
#     np.save('D:/dataset/5x4_Mnist/label_1000_1999_02x2',label_index)
# cv2.namedWindow("synthesis", 0)
# cv2.imshow("synthesis", null_space_5x4)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


