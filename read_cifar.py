import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt

data_dir = "data/"
extract_folder = 'cifar-10-batches-bin'
def encode_labels( y, k):
	"""Encode labels into one-hot representation
	"""
	onehot = np.zeros((y.shape[0],k ))
	for idx, val in enumerate(y):
		onehot[idx,val] = 1.0  ##idx=0~xxxxx，if val =3 ,表示欄位3要設成1.0
	return onehot

def load_train_data(n):			#n=1,2..5,data_batch_1.bin ~data_batch_5.bin
	"""Load Cifar10 data from `path`"""
	images_path = os.path.join(data_dir, extract_folder, 'data_batch_{}.bin'.format(n)) 
	with open(images_path, 'rb') as imgpath:
		images = np.fromfile(imgpath, dtype=np.uint8)
	return images
	
def load_test_data():			#test_batch
	"""Load Cifar10 test data from `path`"""
	test_path = os.path.join(data_dir, extract_folder, 'test_batch.bin') 
	with open(test_path, 'rb') as testpath:
		test_img = np.fromfile(testpath, dtype=np.uint8)
	return test_img	

def display(image,label):
	#Show 50  picture				
	for i in range(10):
		plt.subplot(5,10,i+1),plt.imshow(image[i])
		plt.title(label[i]), plt.xticks([]), plt.yticks([])
		plt.subplot(5,10,i+11),plt.imshow(image[i+10])
		plt.title(label[i+10]), plt.xticks([]), plt.yticks([])
		plt.subplot(5,10,i+21),plt.imshow(image[i+20])
		plt.title(label[i+20]), plt.xticks([]), plt.yticks([])
		plt.subplot(5,10,i+31),plt.imshow(image[i+30])
		plt.title(label[i+30]), plt.xticks([]), plt.yticks([])
		plt.subplot(5,10,i+41),plt.imshow(image[i+40])
		plt.title(label[i+40]), plt.xticks([]), plt.yticks([])	
	plt.show()			


##Load Cifar-10 train image and label	
X_train_image = load_train_data(1)	#load data_batch_1.bin
#reshape to (10000,3073)
#in one Row ,the 1st byte is the label,other 3072byte =1024 Red +1024 green +1024 blue ch data
X_train_image=X_train_image.reshape(-1,3073)
tempA=X_train_image.copy()
X_train_image=np.delete(X_train_image, 0, 1) #delete 1st column data
X_train_image=X_train_image.reshape(-1,3,32,32)  #(1000,3,32,32)
X_train_image = X_train_image.transpose([0, 2, 3, 1])	#transfer to (10000,32,32,3)

#split to 3073 col,the first column is the label.
tempA=np.hsplit(tempA,3073)	
X_train_label=np.asarray(tempA[0])
X_train_label=X_train_label.reshape([10000,])

print(X_train_image.shape)	
print(X_train_label.shape)	
print(X_train_label[0:50])	

display(X_train_image,X_train_label)

##Load Cifar-10 test image  and label	
X_test_image = load_test_data()	#load test_batch.bin
#reshape to (10000,3073)
#in one Row ,the 1st byte is the label,other 3072byte =1024 Red +1024 green +1024 blue ch data
X_test_image=X_test_image.reshape(-1,3073)
tempA=X_test_image.copy()
X_test_image=np.delete(X_test_image, 0, 1) #delete 1st column data
X_test_image=X_test_image.reshape(-1,3,32,32)  #(1000,3,32,32)
X_test_image = X_test_image.transpose([0, 2, 3, 1])	#transfer to (10000,32,32,3)

#split to 3073 col,the first column is the label.
tempA=np.hsplit(tempA,3073)	
X_test_label=np.asarray(tempA[0])
X_test_label=X_test_label.reshape([10000,])

print(X_test_image.shape)	
print(X_test_label.shape)	
print(X_test_label[0:50])	

display(X_test_image,X_test_label)
	
