#Librerias
import nltk
from nltk.tokenize import TweetTokenizer
tok = TweetTokenizer()
import re

# ------------------------------
#        LEER   TRAIN
# ------------------------------

# Leer el archivo de train, cada fila es un tweet que hay que tokenizar
with open ('train.data', 'r') as f:
	data_train = f.readlines()

train = []
for line in data_train:
    sentence = re.split(r'\t+',  line)
    if ( len(sentence) > 2):		# para evitar lineas vacías
        train.append(sentence[2])


# ------------------------------
#        LEER   TEST
# ------------------------------

# Leer el archivo de train, cada fila es un tweet que hay que tokenizar
with open ('test.data', 'r') as f:
	data_test = f.readlines()

test = []
for line in data_test:
	sentence = re.split(r'\t+',  line)
	if ( len(sentence) > 2):		# para evitar lineas vacías
		test.append(sentence[2])


# ------------------------------
#            LIWC
# ------------------------------

# Leer categorias, palabras entre % y %
with open ('LIWC2007_English.dic', 'r') as f:
	cont = False
	dic_liwc = {}
	for line in f:
		if line.startswith('%'):
			if cont == True:
				cont = False
			else:
				cont = True
		elif cont == True:
			l = line.split()
			dic_liwc[l[1]] = l[0]
	f.seek(0);
	cont = True
	posemo_key = dic_liwc['posemo']
	negemo_key = dic_liwc['negemo']
	posemo = {}
	negemo = {}
	for line in f:
		if line.startswith('%'):
			if cont == True:
				cont = False
			else:
				cont = True
		elif cont == True:
			l = line.split()
			for i in l:
				if (i == posemo_key):
					posemo[l[0]] = 1
				if (i == negemo_key):
					negemo[l[0]] = 1


#Tokenizar y obtener el resultado de cada tweet
corpus_liwc=[]		# vector sin normalizar
corpus_liwc_norm=[]	# vector normalizado

for line in train:
	token = tok.tokenize(line)
	positivo = 0
	negativo = 0
	vector_tmp = []
	vector_norm = []
	for i in token:
		if (i in posemo):
			positivo = positivo + posemo[i]
		elif (i in negemo):
			negativo = negativo+ negemo[i]
	vector_tmp.extend([positivo, negativo])
	vector_norm.extend([positivo / len(token), negativo / len(token)])
	corpus_liwc.append( (vector_tmp, line))
	corpus_liwc_norm.append( (vector_norm, line) )

# ------------------------------
#            EMOLEX
# ------------------------------
# Crear diccionarios para cada una de las categorias

with open ('emolex.txt', 'r') as f:
    dic_emo_anger = {}
    dic_emo_anticipation = {}
    dic_emo_disgust = {}
    dic_emo_fear = {}
    dic_emo_joy = {}
    dic_emo_negative = {}
    dic_emo_positive = {}
    dic_emo_sadness = {}
    dic_emo_surprise = {}
    dic_emo_trust = {}
    for line in f:
        l = line.split()
        categoria = l[1]
        if (l[2] == "1"):
            if (categoria == "anger"):
                dic_emo_anger[l[0]] = 1
            elif (categoria == "anticipation"):
                dic_emo_anticipation[l[0]] = 1
            elif (categoria == "disgust"):
                dic_emo_disgust[l[0]] = 1
            elif (categoria == "fear"):
                dic_emo_fear[l[0]] = 1
            elif (categoria == "joy"):
                dic_emo_joy[l[0]] = 1
            elif (categoria == "negative"):
                dic_emo_negative[l[0]] = 1
            elif (categoria == "positive"):
                dic_emo_positive[l[0]] = 1
            elif (categoria == "sadness"):
                dic_emo_sadness[l[0]] = 1
            elif (categoria == "surprise"):
                dic_emo_surprise[l[0]] = 1
            elif (categoria == "trust"):
                dic_emo_trust[l[0]] = 1


# Tokenizar y obtener resultado de cada tweet
train_emolex = []          # vector sin normalizar
train_emolex_norm = []     # vector normalizado

for line in train:
    token = tok.tokenize(line)
    positivo = 0
    negativo = 0
    vector_tmp = []
    vector_norm = []
    for i in token:
        if (i in dic_emo_positive):
            positivo = positivo + dic_emo_positive[i]
        elif (i in dic_emo_negative):
            negativo = negativo + dic_emo_negative[i]
    vector_tmp.extend([positivo, negativo])
    vector_norm.extend([positivo / len(token), negativo / len(token)])
    train_emolex.append(vector_tmp)
    train_emolex_norm.append(vector_norm)
