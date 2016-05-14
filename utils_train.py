# Librerias

from sklearn import svm
from sklearn.metrics import mean_squared_error
import time


# ------------------------------
#   UTILIDADES ENTRENAMIENTOS
# ------------------------------

def entrenar(num_entrenamiento, titulo, vector_train, labels_train, vector_test, labels_test):
    print("ENTRENAMIENTO: ", num_entrenamiento, " - ", titulo)
    clasifier = svm.SVR()
    t0 = time.time()
    clasifier.fit(vector_train, labels_train)
    t1 = time.time()
    prediction = clasifier.predict(vector_test)
    t2 = time.time()
    time_train = t1-t0
    time_predict = t2-t1
    print ('Tamaño del vector (filas x columnas): {} x {} '.format(len(vector_train), len(vector_train[0])))
    print('Time train: {} - Time predict: {}'.format(time_train, time_predict))
    print('Result: {}'.format(mean_squared_error(labels_test, prediction)))
    print("\n")
