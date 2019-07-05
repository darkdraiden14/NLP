#!/usr/bin/python3

import pytesseract
import cv2
import csv
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

# reading image
img= cv2.imread('canvas.png')

# extracting text
extrtext= pytesseract.image_to_string(img,lang='eng')

# writing the text into .csv
#with open('mycsv.csv','wb') as f:
#    writer = csv.writer(f, delimiter=",")
#    for line in extrtext.encode(encoding='UTF-8',errors='strict'):
#        writer.writerow(line)


# splitting th data in words
newdata = [i for i in extrtext.split()]
# checking freq
nlpdata=nltk.FreqDist(newdata)
# pootting graph of top words
nlpdata.plot(20,color='red')

# now removing stopwords
removedata = [i for i in newdata if i.lower() not in stopwords.words('english')]
# checking Freq
nlpremove = nltk.FreqDist(removedata)
#Plotting graph of top words after removal of stopwords
nlpremove.plot(20)
plt.show()
