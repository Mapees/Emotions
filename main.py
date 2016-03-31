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
	print (posemo_key)
	print (negemo_key)
	for line in f:
		if line.startswith('%'):
			if cont == True:
				cont = False
			else:
				cont = True
		elif cont == True:
			l = line.split()
			for i in l:
				print (i)
				if (i == posemo_key):
					posemo[l[0]] = 1
				if (i == negemo_key):
					negemo[l[0]] = 1

			
# Tokenizar train
