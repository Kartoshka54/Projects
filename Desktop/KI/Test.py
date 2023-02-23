#Dieses Programm wird mit Hilfe von Machine Learning feststellen ob eine SMS als Spam klassifiziert wird oder nicht

#Libraries die für das Porgramm verwendet werden:

import numpy as np
#Numpy wird uns dabei behilflich sein Arrays effizienter und schneller auszurechenen.
import pandas as pd
#Panda unterstüzt die library Numpy und ist eine große hilfe im bereich Machine Learning.
import nltk
#Nltk steht für "Natural Language Toolkit" und hilft dem Computer einen Text zu analysieren.
from nltk.corpus import stopwords
#Beinhaltet englische wörter die ignoriert werden können ohne den Satz zu "verändern"
#Stopwords gibt uns quasi eine Liste mit allen Pausen woertern die im englischen vorhanden sind, also muessen wir nicht manuell ein array erstellen und diese woerter aufschreiebn.
import string

#Als Dataset für unser Programm nutze ich die spam.csv Datei die ich von "https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset" gedownloaded habe

#Datei Lesen

df = pd.read_csv(r'C:\Users\W10\Desktop\spam.csv', encoding="ISO-8859-1")
#In diesem Punkt ist die Panda library eine grosse hilfe fuer uns weil sie die Datei "liest".
print(df.columns)
#Beginnen wir mit einer Function/Method
#Diese function wird den Text für den PC lesbarer machen und ünnötige stellen von der Datei entfernen

def filterung(v1):
    keine_Zeich_Setz = [char for char in v1 if char != string.punctuation]
    keine_Zeich_Setz = ''.join(keine_Zeich_Setz)
    #Filtert Zeichensetzung von dem Text

    keine_Pause = [word for word in keine_Zeich_Setz.split()
                   if word.lower() != stopwords.words('english')]

    return keine_Pause
    #Filtert Pausenwörter aus dem englischen wie "as", "is", etc.

#Diese Methode hilft unserer K.I den Text besser zu verarbeiten und ein akkurateres ergebnis zu erzielen



#Tokenization
df['v1'].head().apply(filterung)
#print(df)
#Unsere Machine Learning Library
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
#Jetzt werden wir unsere Texte in der Datei zu einer Matrix konvertieren
sms_bow = CountVectorizer(analyzer=filterung).fit_transform(df['v1'])

#Hier wird die Datei in 2 geteil. 80& fuer Training und 20% um es zu testen
x_train, x_test, y_train, y_test = train_test_split(sms_bow, df['v1'], test_size=0.20, random_state=0) #Unsere verschiedenen variablen um zu testen und trainieren

#Jetz schauen wir uns an wie sich unsere Datei veraendert hat
print(sms_bow.shape)

#Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
klass = MultinomialNB().fit(x_train, y_train)

#Prediction Print
print(klass.predict(x_train))

print(y_train.values)

#Jetzt koennen wir sehen ob unsere Methode Akkurat ist und rasiert wie kanackebart
#Die Tabelle unten hilft uns zu sehen, ob unser Programm sauber arbeitet
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
akku = klass.predict(x_train)
print("Train Resultate: ")
print(classification_report(y_train, akku))
print('Akku:', accuracy_score(y_train, akku))

print('Test Resultate: ')
#Reeeeeeee
akku = klass.predict(x_test)
print(classification_report(y_test, akku))
print('Akku:', accuracy_score(y_test, akku))

#Wie wir sehen, arbeiten unsere Test und Train Funktionen beide mit 100&er genauigkeit.
#Natuerlich kann dieses Resultat bei einem groesseren Datansatz schwanken, weshalb die Genauigkeit auf 99 bis 98% reduziert werden koennte.