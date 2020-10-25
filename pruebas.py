import pandas as pd
import sklearn.model_selection as md
from Preprocesadora import Preprocesadora

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Flatten,Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.layers import LeakyReLU,ReLU
from tensorflow.keras import regularizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import Callback, History, EarlyStopping


df = pd.read_csv("analcatdata_marketing.csv")
"""
dic = {'Cant A':[4,5,1,6,5],'Cant B':[15,36,20,26,None],'Cant C':[1.2,1,0,-1,-0.8],'Sexo':["F","H","F",None,None]
       ,'Cat Altura':["Alto","Bajo","Mediano","Bajo",None],'Edad':[15,16,18,None,None]}
df = pd.DataFrame(dic)
"""

prep = Preprocesadora()
reconstruido, perfomance = prep.reconstruir_por_prediccion(df,columnas_excluir=["X1a"])
#reconstruido = prep.remplazar_no_definidos(df)
reconstruido = prep.obtener_filas_definidas_en_columna(reconstruido,"X1a")

X, Y = prep.dividir_dataframe(reconstruido,["X1a"])
X, _ = prep.aplicar_input_scaling(X)
Y = prep.aplicar_one_hot_encoding(prep.categorizar_columna(Y,"X1a"))
x_t,x_p,y_t,y_p = md.train_test_split(X.values,Y.values,test_size=0.30)

modelo = Sequential()
modelo.add(Dense(18,activation=LeakyReLU(0.3),input_dim=x_t.shape[1],kernel_regularizer=regularizers.l2(0.1)))
modelo.add(Dropout(0.5))
modelo.add(Dense(30,activation=LeakyReLU(0.3),kernel_regularizer=regularizers.l2(0.1)))
modelo.add(Dropout(0.3))
modelo.add(Dense(16,activation=LeakyReLU(0.3),kernel_regularizer=regularizers.l2(0.14)))
modelo.add(Dropout(0.1))
modelo.add(Dense(y_t.shape[1],activation="softmax"))

modelo.compile(optimizer=optimizers.Adam(),loss='categorical_crossentropy',metrics=['accuracy'])
hist = modelo.fit(x_t,y_t,batch_size=32,epochs=300,verbose=0,callbacks=EarlyStopping(monitor='loss',patience=5))
resultado =  modelo.evaluate(x_p, y_p,batch_size=x_p.shape[0])
print(resultado)



