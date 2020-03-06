from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import time

#Importing the DataSet
((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()

def Get_All_Distances(query_desc,images):
    images_and_distance = []
    image = []
    for i in range(len(images)):
        image = images[i]
        image_and_distance = { 'image':image,'distance':np.corrcoef(query_desc,image.flatten())[0][1]} #change descriptor here!
        images_and_distance.append(image_and_distance)

    return images_and_distance



def Show_Results(result,size):
    result = list(result)
    plt.figure(figsize=(15,15))
    for i in range(size):
        image_and_desc = result[i]
        plt.subplot(5,10,i+1)
        plt.xticks([image_and_desc['distance']])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image_and_desc['image'], cmap=plt.cm.binary)
    plt.show()


initial = time.time()

images_descriptors = Get_All_Distances(testX[1].flatten(),trainX)
result = list(filter(lambda x:x['distance'] > 0.6,images_descriptors))

print("Execution Time : ",time.time() - initial)

Show_Results(result,20)

