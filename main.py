#Librerias
from nltk.tokenize import TweetTokenizer
tok = TweetTokenizer()
import re

from sklearn import svm
from sklearn.metrics import classification_report
import time
from sklearn.metrics import mean_squared_error


# ------------------------------
#        LEER   TRAIN
# ------------------------------

# Leer el archivo de train, cada fila es un tweet que hay que tokenizar
with open ('train.data', 'r') as f:
	data_train = f.readlines()

train = []
train_labels = []
for line in data_train:
    sentence = re.split(r'\t+',  line)
    if ( len(sentence) > 2):		# para evitar lineas vacías
        train.append(sentence[2])
        label = sentence[-1]
        label = label[:-1]
        train_labels.append(float(label))


# ------------------------------
#        LEER   TEST
# ------------------------------

# Leer el archivo de train, cada fila es un tweet que hay que tokenizar
with open ('test.data', 'r') as f:
	data_test = f.readlines()
 
test = []
test_labels = []
for line in data_test:
    sentence = re.split(r'\t+',  line)
    if ( len(sentence) > 2):		# para evitar lineas vacías
        test.append(sentence[2])
        label = sentence[1]
        test_labels.append(float(label))



# ------------------------------
#            LIWC
# ------------------------------
# Leer categorias, palabras entre % y %
categorias_liwc = {}
dic_liwc = {}
count_train_liwc = {}  # tiene la forma {categoria, contador}


with open ('LIWC2007_English.dic', 'r') as f:
    cont = False
    for line in f:
        if line.startswith('%'):
            if cont == True:
                cont = False
            else:
                cont = True
        elif cont == True:
            l = line.split()
            categorias_liwc[l[0]] = l[1]
            count_train_liwc[l[1]] = 0
        else:
            l = line.split()
            for i in l[1:-1]:
                dic_aux = {}
                categoria = categorias_liwc[i]
                if categoria in dic_liwc:
                    dic_aux = dic_liwc[categoria]
                dic_aux[l[0]] = 1
                dic_liwc[categoria] = dic_aux
                                
#Tokenizar y obtener el resultado de cada tweet
train_liwc_pn = []            # vector sin normalizar (solo posemo y negemo)
train_liwc_pn_norm = []       # vector normalizado (solo posemo y negemo)
train_liwc = []                # vector completo
train_liwc_norm = []

for line in train:
    token = tok.tokenize(line)
    positivo = 0
    negativo = 0
    vector_tmp = []
    vector_norm = []
    for i in token:
        posemo = dic_liwc['posemo']
        negemo = dic_liwc['negemo']
        if (i in posemo):
            positivo = positivo + posemo[i]
        elif (i in negemo):
            negativo = negativo+ negemo[i]
    vector_tmp.extend([positivo, negativo])
    vector_norm.extend([positivo / len(token), negativo / len(token)])
    train_liwc_pn.append(vector_tmp)
    train_liwc_pn_norm.append(vector_norm)
    
    count_tmp = count_train_liwc.copy()
    for i in token:
        palabra = i.lower()
        for categoria in dic_liwc:
            dic_aux = dic_liwc[categoria]        
            if (palabra in dic_aux):
                valor = count_tmp[categoria]
                valor = valor + dic_aux[palabra]
                count_tmp[categoria] = valor             
            
    vector=[]
    vector_temp_norm = []
    for key in count_tmp:
        vector.append(count_tmp[key])
        vector_temp_norm.append(count_tmp[key]/ len(token))
    train_liwc.append(vector)
    train_liwc_norm.append(vector_temp_norm)
   
test_liwc_pn = []
test_liwc_pn_norm = []
test_liwc = []
test_liwc_norm = []

for line in test:
    token = tok.tokenize(line)
    positivo = 0
    negativo = 0
    vector_tmp = []
    vector_norm = []
    for i in token:
        posemo = dic_liwc['posemo']
        negemo = dic_liwc['negemo']
        if (i in posemo):
            positivo = positivo + posemo[i]
        elif (i in negemo):
            negativo = negativo + negemo[i]
    vector_tmp.extend([positivo, negativo])
    vector_norm.extend([positivo / len(token), negativo / len(token)])
    test_liwc_pn.append(vector_tmp)
    test_liwc_pn_norm.append(vector_norm)
    
    count_tmp = count_train_liwc.copy()
    for i in token:
        palabra = i.lower()
        for categoria in dic_liwc:
            dic_aux = dic_liwc[categoria]        
            if (palabra in dic_aux):
                valor = count_tmp[categoria]
                valor = valor + dic_aux[palabra]
                count_tmp[categoria] = valor             
            
    vector=[]
    vector_temp_norm = []
    for key in count_tmp:
        vector.append(count_tmp[key])
        vector_temp_norm.append(count_tmp[key]/ len(token))
    test_liwc.append(vector)
    test_liwc_norm.append(vector_temp_norm)

# # ------------------------------
# #            EMOLEX
# # ------------------------------
# # Crear diccionarios para cada una de las categorias
#
# with open ('emolex.txt', 'r') as f:
#     dic_emo_anger = {}
#     dic_emo_anticipation = {}
#     dic_emo_disgust = {}
#     dic_emo_fear = {}
#     dic_emo_joy = {}
#     dic_emo_negative = {}
#     dic_emo_positive = {}
#     dic_emo_sadness = {}
#     dic_emo_surprise = {}
#     dic_emo_trust = {}
#     for line in f:
#         l = line.split()
#         categoria = l[1]
#         if (l[2] == "1"):
#             if (categoria == "anger"):
#                 dic_emo_anger[l[0]] = 1
#             elif (categoria == "anticipation"):
#                 dic_emo_anticipation[l[0]] = 1
#             elif (categoria == "disgust"):
#                 dic_emo_disgust[l[0]] = 1
#             elif (categoria == "fear"):
#                 dic_emo_fear[l[0]] = 1
#             elif (categoria == "joy"):
#                 dic_emo_joy[l[0]] = 1
#             elif (categoria == "negative"):
#                 dic_emo_negative[l[0]] = 1
#             elif (categoria == "positive"):
#                 dic_emo_positive[l[0]] = 1
#             elif (categoria == "sadness"):
#                 dic_emo_sadness[l[0]] = 1
#             elif (categoria == "surprise"):
#                 dic_emo_surprise[l[0]] = 1
#             elif (categoria == "trust"):
#                 dic_emo_trust[l[0]] = 1
#
#
# # Tokenizar y obtener resultado de cada tweet
# train_emolex = []          # vector sin normalizar
# train_emolex_norm = []     # vector normalizado
#
# for line in train:
#     token = tok.tokenize(line)
#     positivo = 0
#     negativo = 0
#     vector_tmp = []
#     vector_norm = []
#     for i in token:
#         if (i in dic_emo_positive):
#             positivo = positivo + dic_emo_positive[i]
#         elif (i in dic_emo_negative):
#             negativo = negativo + dic_emo_negative[i]
#     vector_tmp.extend([positivo, negativo])
#     vector_norm.extend([positivo / len(token), negativo / len(token)])
#     train_emolex.append(vector_tmp)
#     train_emolex_norm.append(vector_norm)
#
# # ------------------------------
# #     LIWC    +       EMOLEX
# # ------------------------------
# # Tokenizar y obtener resultado de cada tweet
# train_liwc_emo = []          # vector sin normalizar
# train_liwc_emo_norm = []     # vector normalizado
#
#
# for line in train:
#     token = tok.tokenize(line)
#     positivo_liwc = 0
#     negativo_liwc = 0
#     positivo_emo = 0
#     negativo_emo = 0
#     vector_tmp = []
#     vector_norm = []
#     for i in token:
#         # LIWC
#         if (i in liwc_posemo):
#                 positivo_liwc = positivo_liwc + liwc_posemo[i]
#         elif (i in liwc_negemo):
#                 negativo_liwc = negativo_liwc + liwc_negemo[i]
#         # EMOLEX
#         if (i in dic_emo_positive):
#             positivo_emo = positivo_emo + dic_emo_positive[i]
#         elif (i in dic_emo_negative):
#             negativo_emo = negativo_emo + dic_emo_negative[i]
#     vector_tmp.extend([positivo_liwc, negativo_liwc, positivo_emo, negativo_emo])
#     vector_norm.extend([positivo_liwc / len(token), negativo_liwc / len(token), positivo_emo / len(token), negativo_emo / len(token)])
#     train_liwc_emo.append(vector_tmp)        print(sentence)

#     train_liwc_emo_norm.append(vector_norm)
#

# ------------------------------
#     ENTRENAMIENTO
# ------------------------------
print("OPCIÓN 1: LIWC (posemo y negemo)")
clasifier = svm.SVR()
t0 = time.time()
clasifier.fit(train_liwc_pn_norm, train_labels)
t1 = time.time()
prediction = clasifier.predict(test_liwc_pn_norm)
t2 = time.time()
time_train = t1-t0
time_predict = t2-t1
print('Time train: {} - Time predict: {}'.format(time_train, time_predict))
print(mean_squared_error(test_labels, prediction))

print("OPCIÓN 2: LIWC completo")
clasifier = svm.SVR()
t0 = time.time()
clasifier.fit(train_liwc_norm, train_labels)
t1 = time.time()
prediction = clasifier.predict(test_liwc_norm)
t2 = time.time()
time_train = t1-t0
time_predict = t2-t1
print('Time train: {} - Time predict: {}'.format(time_train, time_predict))
print(mean_squared_error(test_labels, prediction))
