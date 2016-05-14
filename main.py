#Librerias

import utils_corpus
import utils_recursos
import utils_emoticonos
import utils_train

# ------------------------------
#     LEER   TRAIN - TEST
# ------------------------------

# Data
data_train = utils_corpus.obtener_data('train.data');
data_test = utils_corpus.obtener_data('test.data');

# Tuis y labels
train, train_labels = utils_corpus.obtener_tuits_labels_train(data_train)
test, test_labels = utils_corpus.obtener_tuits_labels_test(data_test)

# ------------------------------
#            CORPUS
# ------------------------------

# Obtener corpus tokenizados y sin stopwords
corpus_train_tokenizado = utils_corpus.obtener_corpus_tokenizado(train)
corpus_train_token_stop = utils_corpus.obtener_corpus_stopwords(corpus_train_tokenizado)

corpus_test_tokenizado = utils_corpus.obtener_corpus_tokenizado(test)
corpus_test_token_stop = utils_corpus.obtener_corpus_stopwords(corpus_test_tokenizado)

# ------------------------------
#          RECURSOS
# ------------------------------

# LIWC
diccionario_liwc, vector_inicial_liwc, categorias_liwc = utils_recursos.obtener_diccionario_liwc()

# EMOLEX
diccionario_emolex, vector_inicial_emolex = utils_recursos.obtener_diccionario_emolex()


# ------------------------------
#     VECTORES TOKENIZADOS
# ------------------------------
# Positivo - negativo
vector_train_pn_liwc, vector_train_pn_liwc_norm = utils_recursos.obtener_vector_pn_liwc(corpus_train_tokenizado, diccionario_liwc)
vector_test_pn_liwc, vector_test_pn_liwc_norm = utils_recursos.obtener_vector_pn_liwc(corpus_test_tokenizado, diccionario_liwc)

vector_train_pn_emolex, vector_train_pn_emolex_norm = utils_recursos.obtener_vector_pn_emolex(corpus_train_tokenizado, diccionario_emolex)
vector_test_pn_emolex, vector_test_pn_emolex_norm = utils_recursos.obtener_vector_pn_emolex(corpus_test_tokenizado, diccionario_emolex)

# Completos
vector_train_liwc, vector_train_liwc_norm = utils_recursos.obtener_vector(corpus_train_tokenizado, diccionario_liwc, vector_inicial_liwc)
vector_test_liwc, vector_test_liwc_norm = utils_recursos.obtener_vector(corpus_test_tokenizado, diccionario_liwc, vector_inicial_liwc)

vector_train_emolex, vector_train_emolex_norm = utils_recursos.obtener_vector(corpus_train_tokenizado, diccionario_emolex, vector_inicial_emolex)
vector_test_emolex, vector_test_emolex_norm = utils_recursos.obtener_vector(corpus_test_tokenizado, diccionario_emolex, vector_inicial_emolex)

# ------------------------------
#  VECTORES TOKENIZADOS + STOP
# ------------------------------
# Positivo - negativo
vector_train_pn_liwc_stop, vector_train_pn_liwc_norm_stop = utils_recursos.obtener_vector_pn_liwc(corpus_train_token_stop, diccionario_liwc)
vector_test_pn_liwc_stop, vector_test_pn_liwc_norm_stop = utils_recursos.obtener_vector_pn_liwc(corpus_test_token_stop, diccionario_liwc)

vector_train_pn_emolex_stop, vector_train_pn_emolex_norm_stop = utils_recursos.obtener_vector_pn_emolex(corpus_train_token_stop, diccionario_emolex)
vector_test_pn_emolex_stop, vector_test_pn_emolex_norm_stop = utils_recursos.obtener_vector_pn_emolex(corpus_test_token_stop, diccionario_emolex)

# Completos
vector_train_liwc_stop, vector_train_liwc_norm_stop = utils_recursos.obtener_vector(corpus_train_token_stop, diccionario_liwc, vector_inicial_liwc)
vector_test_liwc_stop, vector_test_liwc_norm_stop = utils_recursos.obtener_vector(corpus_test_token_stop, diccionario_liwc, vector_inicial_liwc)

vector_train_emolex_stop, vector_train_emolex_norm_stop = utils_recursos.obtener_vector(corpus_train_token_stop, diccionario_emolex, vector_inicial_emolex)
vector_test_emolex_stop, vector_test_emolex_norm_stop = utils_recursos.obtener_vector(corpus_test_token_stop, diccionario_emolex, vector_inicial_emolex)

#
# ------------------------------
#     VECTORES ENTRENAMIENTO
# ------------------------------
# Tokenizados
# LIWC + EMOLEX (positivo y negativo)
vector_train_pn_liwc_emo = utils_recursos.duplicar_vectores(vector_train_pn_liwc, vector_train_pn_emolex)
vector_test_pn_liwc_emo = utils_recursos.duplicar_vectores(vector_test_pn_liwc, vector_test_pn_emolex)

vector_train_pn_liwc_emo_norm = utils_recursos.duplicar_vectores(vector_train_pn_liwc_norm, vector_train_pn_emolex_norm)
vector_test_pn_liwc_emo_norm = utils_recursos.duplicar_vectores(vector_test_pn_liwc_norm, vector_test_pn_emolex_norm)

# LIWC + EMOLEX (completo)
vector_train_liwc_emo = utils_recursos.duplicar_vectores(vector_train_liwc, vector_train_emolex)
vector_train_liwc_emo_norm = utils_recursos.duplicar_vectores(vector_train_liwc_norm, vector_train_emolex_norm)

vector_test_liwc_emo = utils_recursos.duplicar_vectores(vector_test_liwc, vector_test_emolex)
vector_test_liwc_emo_norm = utils_recursos.duplicar_vectores(vector_test_liwc_norm, vector_test_emolex_norm)

# Tokenizados + stopwords
# LIWC + EMOLEX (positivo y negativo)
vector_train_pn_liwc_emo_stop = utils_recursos.duplicar_vectores(vector_train_pn_liwc_stop, vector_train_pn_emolex_stop)
vector_test_pn_liwc_emo_stop = utils_recursos.duplicar_vectores(vector_test_pn_liwc_stop, vector_test_pn_emolex_stop)

vector_train_pn_liwc_emo_norm_stop = utils_recursos.duplicar_vectores(vector_train_pn_liwc_norm_stop, vector_train_pn_emolex_norm_stop)
vector_test_pn_liwc_emo_norm_stop = utils_recursos.duplicar_vectores(vector_test_pn_liwc_norm_stop, vector_test_pn_emolex_norm_stop)

# LIWC + EMOLEX (completo)
vector_train_liwc_emo_stop = utils_recursos.duplicar_vectores(vector_train_liwc_stop, vector_train_emolex_stop)
vector_train_liwc_emo_norm_stop = utils_recursos.duplicar_vectores(vector_train_liwc_norm_stop, vector_train_emolex_norm_stop)

vector_test_liwc_emo_stop = utils_recursos.duplicar_vectores(vector_test_liwc_stop, vector_test_emolex_stop)
vector_test_liwc_emo_norm_stop = utils_recursos.duplicar_vectores(vector_test_liwc_norm_stop, vector_test_emolex_norm_stop)

# ------------------------------
#         EMOTICONOS
# ------------------------------
# Train
vector_inicial_train_emoticonos = utils_emoticonos.obtener_diccionario_emoticonos(corpus_train_tokenizado)
vector_emoticonos_train, vector_emoticonos_train_norm = utils_emoticonos.obtener_vector(corpus_train_tokenizado, vector_inicial_train_emoticonos)

vector_inicial_train_emoticonos_stop = utils_emoticonos.obtener_diccionario_emoticonos(corpus_train_token_stop)
vector_emoticonos_train_stop, vector_emoticonos_train_norm_stop = utils_emoticonos.obtener_vector(corpus_train_token_stop, vector_inicial_train_emoticonos_stop)

# Test
vector_inicial_test_emoticonos = utils_emoticonos.obtener_diccionario_emoticonos(corpus_test_tokenizado)
vector_emoticonos_test, vector_emoticonos_test_norm = utils_emoticonos.obtener_vector(corpus_test_tokenizado, vector_inicial_train_emoticonos)

vector_inicial_test_emoticonos_stop = utils_emoticonos.obtener_diccionario_emoticonos(corpus_test_token_stop)
vector_emoticonos_test_stop, vector_emoticonos_test_norm_stop = utils_emoticonos.obtener_vector(corpus_test_token_stop, vector_inicial_train_emoticonos_stop)

# VECTORES ENTRENAMIENTO
# Tokenizado
vector_emoticonos_completo_train_norm = utils_recursos.duplicar_vectores(vector_emoticonos_train_norm, vector_train_liwc_emo_norm)
vector_emoticonos_completo_test_norm = utils_recursos.duplicar_vectores(vector_emoticonos_test_norm, vector_test_liwc_emo_norm)

# Tokenizado + stopwords
vector_emoticonos_completo_train_norm_stop = utils_recursos.duplicar_vectores(vector_emoticonos_train_norm_stop, vector_train_liwc_emo_norm_stop)
vector_emoticonos_completo_test_norm_stop = utils_recursos.duplicar_vectores(vector_emoticonos_test_norm_stop, vector_test_liwc_emo_norm_stop)

# ------------------------------
#         BAG OF WORDS
# ------------------------------


bag_train = utils_recursos.obtener_vector_bag_fit_transform(corpus_train_tokenizado)
bag_train_stop = utils_recursos.obtener_vector_bag_fit_transform(corpus_train_token_stop)
        
bag_test = utils_recursos.obtener_vector_bag_fit_transform(corpus_test_tokenizado)
bag_test_stop = utils_recursos.obtener_vector_bag_fit_transform(corpus_test_token_stop)

#print ('Tr: {} x {} '.format(len(bag_train), len(bag_train[0]))) ## 8000 x 18417 
#print ('Te: {} x {} '.format(len(bag_test), len(bag_test[0])))  ## 4000 x 11196


# ------------------------------
#         ENTRENAMIENTOS
# ------------------------------

#print("TOKENIZADOS")
#num_entrenamiento = 1
#titulo = "LIWC (posemo y negemo)"
#utils_train.entrenar(num_entrenamiento, titulo, vector_train_pn_liwc_norm, train_labels, vector_test_pn_liwc_norm, test_labels)
#
#num_entrenamiento = 2
#titulo = "LIWC (completo)"
#utils_train.entrenar(num_entrenamiento, titulo, vector_train_liwc_norm, train_labels, vector_test_liwc_norm, test_labels)
#
#num_entrenamiento = 3
#titulo = "EMOLEX (positive y negative)"
#utils_train.entrenar(num_entrenamiento, titulo, vector_train_pn_emolex_norm, train_labels, vector_test_pn_emolex_norm, test_labels)
#
#num_entrenamiento = 4
#titulo = "EMOLEX (completo)"
#utils_train.entrenar(num_entrenamiento, titulo, vector_train_emolex_norm, train_labels, vector_test_emolex_norm, test_labels)
#
#num_entrenamiento = 5
#titulo = "LIWC + EMOLEX (positivos y negativos)"
#utils_train.entrenar(num_entrenamiento, titulo, vector_train_pn_liwc_emo_norm, train_labels, vector_test_pn_liwc_emo_norm, test_labels)
#
#num_entrenamiento = 6
#titulo = "LIWC + EMOLEX (completo)"
#utils_train.entrenar(num_entrenamiento, titulo, vector_train_liwc_emo_norm, train_labels, vector_test_liwc_emo_norm, test_labels)

# -----------------------------------------

#print("TOKENIZADOS + STOPWORDS")
#num_entrenamiento = 7
#titulo = "LIWC (posemo y negemo)"
#utils_train.entrenar(num_entrenamiento, titulo, vector_train_pn_liwc_norm_stop, train_labels, vector_test_pn_liwc_norm_stop, test_labels)
#
#num_entrenamiento = 8
#titulo = "LIWC (completo)"
#utils_train.entrenar(num_entrenamiento, titulo, vector_train_liwc_norm_stop, train_labels, vector_test_liwc_norm_stop, test_labels)
#
#num_entrenamiento = 9
#titulo = "EMOLEX (positive y negative)"
#utils_train.entrenar(num_entrenamiento, titulo, vector_train_pn_emolex_norm_stop, train_labels, vector_test_pn_emolex_norm_stop, test_labels)
#
#num_entrenamiento = 10
#titulo = "EMOLEX (completo)"
#utils_train.entrenar(num_entrenamiento, titulo, vector_train_emolex_norm_stop, train_labels, vector_test_emolex_norm_stop, test_labels)
#
#num_entrenamiento = 11
#titulo = "LIWC + EMOLEX (positivos y negativos)"
#utils_train.entrenar(num_entrenamiento, titulo, vector_train_pn_liwc_emo_norm_stop, train_labels, vector_test_pn_liwc_emo_norm_stop, test_labels)
#
#num_entrenamiento = 12
#titulo = "LIWC + EMOLEX (completo)"
#utils_train.entrenar(num_entrenamiento, titulo, vector_train_liwc_emo_norm_stop, train_labels, vector_test_liwc_emo_norm_stop, test_labels)

#print("EMOTICONOS")
#num_entrenamiento = 13
#titulo = "EMOTICONOS + Recursos tokenizados"
#utils_train.entrenar(num_entrenamiento, titulo, vector_emoticonos_completo_train_norm, train_labels, vector_emoticonos_completo_test_norm, test_labels)
#
#num_entrenamiento = 14
#titulo = "EMOTICONOS + Recursos tokenizados + stopwords"
#utils_train.entrenar(num_entrenamiento, titulo, vector_emoticonos_completo_train_norm_stop, train_labels, vector_emoticonos_completo_test_norm_stop, test_labels)

print("BAG OG WORDS")
num_entrenamiento = 15
titulo = "BAG OF WORDS + Data tokenizado"
utils_train.entrenar(num_entrenamiento, titulo, bag_train, train_labels, bag_test, test_labels)

#num_entrenamiento = 16
#titulo = "BAG OF WORDS + Data tokenizados + stopwords"
#utils_train.entrenar(num_entrenamiento, titulo, bag_train_stop, train_labels, bag_test_stop, test_labels)
