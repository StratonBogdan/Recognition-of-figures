#Clasificatorul Naive Bayes

import numpy as np
import matplotlib.pyplot as plt



dirPath = "/Users/bogdan/Downloads/data/"
#incarca imaginile de antrenare
train_images = np.loadtxt(dirPath + "train_images.txt")
print(train_images.shape)
print(train_images.ndim)
print(type(train_images[0,0]))
print(train_images.size)
print(train_images.nbytes)


#afiseaza prima imagine
image = train_images[0,:]
image = np.reshape(image,(28,28))
#afiseaza valorile pixelilor
print(image)
#afiseaza imaginea
plt.imshow(image,cmap = "gray")
plt.show()


train_labels = np.loadtxt(dirPath + "train_labels.txt",'int8')
print(train_labels.shape)
print(type(train_labels[0]))
print(train_labels.size)
print(train_labels.nbytes)
#afiseaza eticheta primului exemplu de antrenare
print(train_labels[0])


nbImages = 10
plt.figure(figsize=(5,5))
for i in range(nbImages**2):
    plt.subplot(nbImages,nbImages,i+1)
    plt.axis('off')
    plt.imshow(np.reshape(train_images[i,:],(28,28)),cmap = "gray")
plt.show()
labels_nbImages = train_labels[:nbImages**2]
print(np.reshape(labels_nbImages,(nbImages,nbImages)))



p_C = np.zeros(10,'uint8')
for label in train_labels:
    p_C[label] = p_C[label]+1
print(p_C/sum(p_C))



get_ipython().run_line_magic('pinfo', 'np.histogram')



dateExemplu = np.array([0, 63, 63, 64, 65, 128, 255, 15])
numarIntervale = 4
limiteInterval = np.linspace(0,256,numarIntervale+1)
print(limiteInterval)
histograma,trash = np.histogram(dateExemplu,limiteInterval)
print(histograma)



i = 0
j = 370
#gaseste indecsii imaginilor de antrenare cu cifra 0
index = np.ravel(np.where(train_labels == i))
print("Imaginile de antrenare care contin clasa " + str(i) + " au indecsii:")
print(index)

valoriPixeli = train_images[index,j]
print("Pixelii de pe pozitia " + str(j) + " au valorile:")
print(valoriPixeli)
numarIntervale = 4
limiteInterval = np.linspace(0,256,numarIntervale+1)
print("Limitele intervalelor sunt: " + str(limiteInterval))
histograma,trash = np.histogram(valoriPixeli,limiteInterval)
print("Numarul de valori din fiecare interval este: " + str(histograma))
M_ij = histograma/sum(histograma) 
print("vectorul de probabilitati pentru clasa " + str(i) + " si pozitia " + str(j) + " este: " + str(M_ij))





test_images = np.loadtxt(dirPath + "test_images.txt")
#afiseaza informatii despre imaginile de testare
print(test_images.shape)
print(test_images.ndim)
print(type(test_images[0,0]))
print(test_images.size)
print(test_images.nbytes)



get_ipython().run_line_magic('pinfo', 'np.digitize')

dateExemplu = np.array([0, 63,63, 128, 255, 15])
numarIntervale = 4
limiteInterval = np.linspace(0,256,numarIntervale+1)
print(limiteInterval)
indexInterval = np.digitize(dateExemplu,limiteInterval)
print(indexInterval)



