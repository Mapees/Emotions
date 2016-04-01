#Librerias
import nltk
from nltk.tokenize import TweetTokenizer
tok = TweetTokenizer()
import re

#LIWC
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

			

# Leer el archivo de train, cada fila es un tweet que hay que tokenizar
with open ('train.data', 'r') as f:
	data_liwc = f.readlines()

train = []
for line in data_liwc:
	sentence = re.split(r'\t+',  line)
	if ( len(sentence) > 2):		# para evitar lineas vacías
		train.append(sentence[2])

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
