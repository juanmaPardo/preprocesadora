#Bibliotecas estructuras de datos
import pandas as pd
import numpy as np

#Clases utiles
from sklearn import preprocessing
from collections import Counter

#Funciones utiles
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split

#Biblioteca redes neuronales.
import tensorflow as tf

# Keras stuff
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Flatten,Dropout
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import Callback, History, EarlyStopping

class Preprocesadora:
    """
    Clase que tiene el objetivo de preprocesar un dataframe con el objetivo que el mismo
    pueda ser usado como input para una red neuronal.
    """

    def aplicar_one_hot_encoding(self,df):
        """
        Dado un dataframe, performa un OneHotEncoding sobre las columnas categoricas del dataframe y devu
        y retorna los resultados.
        :param df: Pandas dataframe
        :return One hot encoded dataframe
        """
        copy = df.copy()
        copy = pd.get_dummies(copy)
        return copy

    def __is_dummy__(self,df,col_name):
        """
        La idea es la siguiente, si chekeamos 30 valores que sean distintos a Nan en una columna
        y los 30 son valores que son iguales a 0 o 1 podemos asumir que es una dummy variable. De
        esta manera nos ahorramos recorrer todas las filas
        :param df = Instancia valida de un dataframe
        :param col_name: Columna donde hacemos el chekeo
        :return: True si es una dummy variable(solo unos y ceros) False en caso contrario
        """
        long_dataframe = len(df.index)
        longitud_chekeo = long_dataframe if long_dataframe < 30 else 30
        i = 0
        if(not is_numeric_dtype(df[col_name])):
            return False
        while i < longitud_chekeo:
            if (np.isnan(i)):
                longitud_chekeo += 1 if (longitud_chekeo+1) < long_dataframe else 0
            if(df.at[i,col_name] != 0 and df.at[i,col_name] != 1):
                return False
            i += 1
        return True

    def aplicar_input_scaling(self,df):
        """
        Escala el input para que los mismos se encuentren en una misma escala y se
        facilite la convergencia/disminusca la varianza.
        :param df: Dataframe a escalar
        :return Dataframe con el input a escalado y un diccionario en donde las key son
        las columnas transformadas y el value es una funcion para devolverlas al estado en el
        que se encontraban.
        """
        def estandarizar_valores(df,col):
            """
            Normaliza los valores de la columna a una distribucion normal con media 0 y varianza 1
            :param df: Dataframe
            :param col: Nombre de columna a normalizar
            :return: Valores de la columna normalizados y la funcion necesaria para hacer la
            transformacion inversa y volver al estado original
            """
            valores = df.loc[:,col].values.reshape(-1,1)
            estandarizador = preprocessing.StandardScaler().fit(valores)
            return estandarizador.transform(valores), estandarizador.inverse_transform
        def normalizar_valores(df,col):
            """
            Estandariza los valores de la columna haciendo uso del maximo y minimo.
            :param df: Dataframe
            :param col: Nombre de columna a estandarizar
            :return: Valores de la columna estandarizadops y la funcion necesaria para hacer la
            transformacion inversa y volver al estado original
            """
            valores = df.loc[:, col].values.reshape(-1, 1)
            normalizador = preprocessing.MinMaxScaler().fit(valores)
            return normalizador.transform(valores), normalizador.inverse_transform

        copy = df.copy()
        skewness_test = copy.skew(0)
        dic_inv_transformations = {}
        for col in skewness_test.index:
            is_dummy_variable = self.__is_dummy__(copy,col)
            if(not is_dummy_variable):
                if(skewness_test[col] <= 0.5 and skewness_test[col] >= -0.5):
                    copy[col], inverse_transf_pointer = estandarizar_valores(copy,col)
                else:
                    copy[col], inverse_transf_pointer = normalizar_valores(copy, col)
                dic_inv_transformations[col] = inverse_transf_pointer

        return copy,dic_inv_transformations

    def deshacer_input_scaling(self,df,dic_inversiones):
        """
        Deshace la transformacion de input scaling aplicada al dataframe
        :param df = Dataframe
        :param dic_inversiones = Dictionario cuyos valor es la funcion para devolver a cada
        columna a la normalidad
        :return = Dataframe con las columnas devueltas a su estado pre-transformacion
        """
        copy = df.copy()
        for key,inv_trans in dic_inversiones.items():
            copy[key] = inv_trans(copy[key].values.reshape(-1,1))

        return copy

    def discretizar_valores_columna(self,df,col):
        """
        Transforma el tipo de dato de una columna a entero. La idea es pasar una columna de
        tipo flotante a tipo entero de ser necesario.
        :param df = Instancia valida de un dataframe
        :param col = Nombre de la columna a convertir
        :return = Dataframe en donde se discretizo la columna indicada
        """
        copy = df.copy()
        if is_numeric_dtype(copy[col]) and all(pd.notnull(copy[col].values)):
            copy[col] = [int(valor) for valor in copy[col].values]
        return copy

    def __es_columna_discreta__(self,df,target_column):
        """
        Devuelve True si todos los valroes definidos de la columna son discretos
        :param df: Instancia valida de un dataframe
        :param target_column: Nombre de la columna
        :return: Devuelve True si todos los valores de la columna son discretos, False
        en caso contrario
        """
        not_null_values = df[target_column].values[pd.notnull(df[target_column].values)]
        return all([int(valor)==valor for valor in not_null_values])

    def pasar_a_flotante(self,df,col):
        """
        Transforma el tipo de dato de una columna a flotante. La idea es pasar una columna de
        tipo entero a tipo flotante de ser necesario.
        :param df = Instancia valida de un dataframe
        :param col = Nombre de la columna a convertir
        :return = Dataframe en donde los valores de la columna indicada ahora son flotantes
        """
        copy = df.copy()
        if is_numeric_dtype(copy[col]) and all(pd.notnull(copy[col].values)):
            copy[col] = [float(valor) for valor in copy[col].values]
        return copy

    def remplazar_numericos_no_definidos(self,df,col,tipo_remplazo="mean"):
        """
        Dada una columna y un tipo de remplazo, remplaza los valores no definidos por el tipo
        de remplazo indicado, siendo los tipos posibles el remplazo por mean o median.
        :param df = Instancia valida de un dataframe
        :param col: Nombre de la columna en el dataframe
        :param tipo_remplazo: Tipo de remplazo a realizar 'mean'/'median'
        :return = Dataframe en donde los valores numericos no definidos fueron remplazados
        de la manera solicitada
        """
        copy = df.copy()
        if is_numeric_dtype(copy[col]):
            definidos = copy[col].values[pd.notnull(copy[col].values)]
            if tipo_remplazo == "mean":
                son_discretos = np.array_equal(definidos, definidos.astype(int))
                mean = definidos.mean()
                if son_discretos:#Si la columna es de tipo discreto
                    mean = int(mean)
                copy[col].fillna(mean,inplace=True)
            elif tipo_remplazo == "median":
                copy[col].fillna(copy[col].median(), inplace=True)
            if(son_discretos):
                copy = self.discretizar_valores_columna(copy, col)
        return copy

    def remplazar_categoricos_no_definidos(self,df,col,most_frecuent=True):
        """
        Dado una columna y un tipo de remplazo, remplaza los valores no definidos por ese
        tipo de remplazoo, siendo los dos tipos posibles por el valor mas frecuente, o directamente
        una nueva categoria para los valores no definidos.
        :param df = Instancia valida de un dataframe
        :param col: Nombre de la columna
        :param most_frecuent: True en caso que se quiera remplazar los valores indefinidos por la categoria
        mas frecuente, False en caso que se cree una nueva categoria con el nombre "Undefined".
        :return = Dataframe en donde se remplazaron los valores categoricos no definidos
        """
        copy = df.copy()
        if is_string_dtype(copy[col]):
            if most_frecuent:
                contador = Counter(copy[col])
                mas_frecuente = contador.most_common(1)[0][0]
                copy[col].fillna(mas_frecuente,inplace=True)
            else:
                copy[col].fillna("Undefined", inplace=True)
        return copy

    def remplazar_no_definidos(self,df):
        """
        La idea de esta funcion es de remplazar de manera rapida todos los valores no definidos
        en el dataframe, de manera tal que si la columna es de tipo numerico se remplaza por la mean,
        y si la misma es de tipo categorico se remplaza por la categoria mas frecuente.
        En caso de que se nececite ser mas especifico usar las otras funciones.
        :param df = Instancia valida de un dataframe
        :return = Dataframe en donde los valores no definidos fueron remplazados.
        """
        copy = df.copy()
        for col in copy:
            if is_numeric_dtype(copy[col]):
                copy = self.remplazar_numericos_no_definidos(copy,col)
            else:
                copy = self.remplazar_categoricos_no_definidos(copy,col)
        return copy

    def __obtener_training_df__(self,df,target_column):
        """
        Dado un dataframe y una columna target devuelve un dataframe el cual:
            * Se remplazaron los valores no definidos en las columnas numericas por la
              mean, y por el mas frecuente en las columnas de tipo categorico. Este
              proceso excluye la target column ya que esa la vamos a predecir con una
              red neuronal feed-forward.
            * Se aplico una hot encoding sobre las columnas categoricas
            * Se applico input scaling sobre las columnas de tipo numerico
        :param df: Instancia de dataframe valido
        :param target_column: Columna target
        :return: Dataframe en donde las columnas estan listas para ser usadas como entrenamiento
        para una red neuronal, junto a un diccionario para revertir el input scaling
        """
        training_df_input_escalado, dic_inversas = self.aplicar_input_scaling(df)
        training_df_input_not_null = self.remplazar_no_definidos(training_df_input_escalado.drop(target_column,1))
        training_df_one_hot_encoded = self.aplicar_one_hot_encoding(training_df_input_not_null)
        training_df_one_hot_encoded[target_column] = training_df_input_escalado[target_column].values
        return training_df_one_hot_encoded,dic_inversas

    def __calcular_cantidad_de_categorias__(self,df,target_column):
        """
        Calcula la cantidad de categorias unicas en una columna.
        :param df: Instancia de dataframe valido
        :param target_column: Columna de tipo categorico
        :return: Entero representando la cantidad de categorias en la target_column
        """
        return len(np.unique(df[target_column].values))

    def __calcular_neuronas_hidden_layer__(self,cant_inputs,factor_phl,factor_crecimiento,
                                        factor_decrecimiento,depth):
        """
        Devuelve una lista representando cuantas neuronas deberia haber por capa. Obviamente no
        es lo ideal, es una generalizacion, la idea es que el pico de neuronas este en la capa
        intermedia, y de ahi decrezca en forma de escalera hacia el output. En realidad lo ideal
        seria probar muchas redes neuronales y ver cual predice mejor, pero el objetivo de esto
        no es obtener la mejor prediccion del mundo sino recuperar los valores del dataframe con
        valores mas precisos que simplemente remplazar por la mean o el mas frecuente.
        :param cant_inputs: Cantidad de neuronas en la input layer
        :param factor_phl: (Factor_primerahiddenlayer) que representa el valor por el cual se va a multiplicar la cantidad
        de neuronas del input para definir cuantas neuronas habra en la primera capa. Se recomienda
        que dicho factor sea 2/3 como maximo.
        :param factor_crecimiento: Es el factor que va a determinar el ratio de crecimiento, es decir
        cuanto van a crecer la cantidad de neuronas por capa hasta llegar a una peak. La idea es que sea
        mayor a uno
        :param factor_decrecimiento: Es el factor que va a determinar el ratio de decrecimiento, es decir
        cuanto van a deecrecer la cantidad de neuronas por capa hasta despues de la peak. La idea es que sea menor a uno
        :param depth: Cantidad de capas que deseamos que tenga la red neuronal.
        :return: Devuelve una lista representando cuantas neuronas deberia haber por capa
        """
        def calculate_peak(depth):
            """
            Calcula las posiciones en donde se va a encontrar el pico de neuronas.
            :param depth: Profundidad de la red neuronal
            :return: Lista que representa en que capa/capas se va a encontrar el pico de neuronas
            """
            mitad_depth = depth / 2
            if depth%2 == 0:
                return [mitad_depth-1,mitad_depth]
            return [int(mitad_depth)]#int(num.5) = num
        cant_neuronas_layer = []
        peak = calculate_peak(depth)
        for i in range(depth):
            if i==0:#Primera capa
                cant_neuronas_layer.append(int(cant_inputs*factor_phl))
            elif i < peak[0]:#Capas previas a la peak
                cant_neuronas_layer.append(int(cant_neuronas_layer[i-1]*factor_crecimiento))
            elif all([i > peak_layer for peak_layer in peak]):#Capas post-peak
                cant_neuronas_layer.append(int(cant_neuronas_layer[i-1]*factor_decrecimiento))
            elif i in peak:#Capas peak
                if i == peak[0]:
                    cant_neuronas_layer.append(int(cant_neuronas_layer[i - 1] * factor_crecimiento))
                else:
                    cant_neuronas_layer.append(cant_neuronas_layer[i-1])
        return cant_neuronas_layer

    def __generar_modelo__(self,df_shape,neuronas_output,factor_phl=0.58,factor_crecimiento=1.8,
                           factor_decrecimiento=0.7,fact="relu",depth=3):
        """
        Genera un modelo secuencial que se utilizara para entrenar la red neuronal.
        :param df_shape: Shape del dataframe
        :param neuronas_output: Cantidad de neuronas que tiene el output
        :param factor_phl: (Factor_primerahiddenlayer) que representa el valor por el cual se va a multiplicar la cantidad
        de neuronas del input para definir cuantas neuronas habra en la primera capa. Se recomienda
        que dicho factor sea 2/3 como maximo.
        :param factor_crecimiento: Es el factor que va a determinar el ratio de crecimiento, es decir
        cuanto van a crecer la cantidad de neuronas por capa hasta llegar a una peak. La idea es que sea
        mayor a uno
        :param factor_decrecimiento: Es el factor que va a determinar el ratio de decrecimiento, es decir
        cuanto van a deecrecer la cantidad de neuronas por capa hasta despues de la peak. La idea es que sea menor a uno
        :param fact: Funcion de activacion a utilizar
        :param depth: Cantidad de capas que deseamos que tenga la red neuronal.
        :return: Modelo sequential instanciado de acuerdo a los parametros indicados.
        """
        hidden_layers = []
        cant_neuronas_layer = self.__calcular_neuronas_hidden_layer__(df_shape[1],factor_phl,factor_crecimiento,factor_decrecimiento,depth)
        #Instanciamos hidden layers
        for i in range(depth):
            if i == 0:
                hidden_layers.append(Dense(cant_neuronas_layer[i],activation=fact,input_dim=df_shape[1],kernel_regularizer=regularizers.l2(0.1)))
            else:
                hidden_layers.append(Dense(cant_neuronas_layer[i], activation=fact,kernel_regularizer=regularizers.l2(0.086)))
        #Instanciamos output layer
        if neuronas_output == 1:
            hidden_layers.append(Dense(1))
        else:
            hidden_layers.append(Dense(neuronas_output,activation="softmax"))

        #Creamos modelo
        modelo = Sequential()
        for layer in hidden_layers:
            modelo.add(layer)
            modelo.add(Dropout(np.random.rand()*0.5))#Agregamos dropout con probabilidad random
        return modelo

    def __obtener_prediccion__(self,df,X_predecir,target_column):
        """
        Crea una red neuronal con el objetivo de utilizar las samples contenidas en el dataframe
        enviado por parametro para aprender de como a travez de las mismas podemos predecir la target_column.
        El objetivo es entrenar la red neuronal y luego utilizarla para predecir el input enviado
        en la variable X_predecir
        :param df: Dataframe valido para ingresar como input a la red neuronal
        :param X_predecir: X que queremos predecir una vez la red neuronal este entrenada
        :param target_column: Columna objetivo, la cual sera el output de nuestra red neuronal
        :return: Retorna un numpy array representando el output que la red neuronal predijo para
        el input X_predecir.
        """
        input = df.drop(target_column,1).values
        output = df[target_column].values.reshape(-1,1)
        if is_numeric_dtype(df[target_column]):
            output_neurons = 1
            model = self.__generar_modelo__(input.shape,output_neurons)
            model.compile(optimizer="adam",loss=losses.mean_squared_error,metrics=[metrics.mae])
            history = model.fit(input,output,batch_size=32,epochs=300,verbose=0,callbacks=EarlyStopping(monitor='loss',patience=5))
            return model.predict(X_predecir), history.history['mean_absolute_error']
        else:
            output_neurons = self.__calcular_cantidad_de_categorias__(df,target_column)
            model = self.__generar_modelo__(input.shape,output_neurons)
            model.compile(optimizer="adam",loss="categorical_crossentropy", metrics=['accuracy'])
            history = model.fit(input,output,batch_size=32,epochs=300,verbose=0,callbacks=EarlyStopping(monitor='loss',patience=5))
            return model.predict(X_predecir), history.history['accuracy']

    def __rellenear_no_definidos_by_lista__(self,df,nombre_columna,missing_values):
        """
        Dado un df, una columna y una lista, utiliza los valores de la lista para
        remplazar cada uno de los valores no definidos en el dataframe en la columna 'column'
        :param df: Instancia valida de un dataframe
        :param column: Nombre de la columna
        :param lista_valores: Lista que contiene tantos valores como valores no se encuentran deefinidos
        en la columna 'column'
        :return: Numpy array representando los values de la columna pos-rellenamiento
        """
        copy = df.copy()
        i_missing = 0
        for indice in range(copy.shape[0]):
            value_at_indice = df.at[indice,nombre_columna]
            copy.at[indice,nombre_columna] = missing_values[i_missing] if np.isnan(value_at_indice) else value_at_indice
            i_missing += 1 if np.isnan(value_at_indice) else 0
        return copy[nombre_columna].values

    def __predecir_no_definidos__(self,df,target_column):
        """
        Devuelve un numpy array que contiene los valores predichos que no se encuentran
        definidos en la target_column.
        :param df: Instancia valida de un dataframe
        :param target_column: Columna a predecir
        :return: Numpy array representando los values de la target_column pos-prediccion
        """
        copy = df.copy()
        not_null_values = pd.notnull(copy[target_column].values)
        df_preparado_para_entrenamiento, dic_inv = self.__obtener_training_df__(copy, target_column)
        training_df = df_preparado_para_entrenamiento[not_null_values]#.reset_index().drop("index")
        df_input_a_predecir = df_preparado_para_entrenamiento[[not i for i in not_null_values]]
        X_predecir = df_input_a_predecir.drop(target_column, 1).values
        prediccion, goodness_of_fit = self.__obtener_prediccion__(training_df, X_predecir, target_column)
        df_preparado_para_entrenamiento[target_column] = self.__rellenear_no_definidos_by_lista__(df_preparado_para_entrenamiento,target_column,prediccion)
        df_target_column_reconstruida = self.deshacer_input_scaling(df_preparado_para_entrenamiento,dic_inv)
        if(self.__es_columna_discreta__(copy,target_column)):
            df_target_column_reconstruida = self.discretizar_valores_columna(df_target_column_reconstruida,target_column)
        return df_target_column_reconstruida[target_column].values,goodness_of_fit

    def reconstruir_por_prediccion(self,df,columnas_excluir=[]):
        """
        Esta funcionalidad tiene el objetivo de crear una red neoronal feed-forward por
        cada categoria que tenga datos no definidos, en donde dicha columna se va a interpretar como
        el output, y se va a utilizar las samples del resto de las columnas para aprender a como
        el valor de dicho input puede ser predicho por el valor del resto de los inputs, de manera
        tal que una vez la red neuronal este entrenada podamos usarla para predecir los valores
        que no se encuentran definidos en dicha columna haciendo uso de esa red neuronal.
        :param df = Dataframe que queramos reconstruir
        :param columnas_excluir = Lista que representa las columnas que queremos excluir en
        la recomposicion
        :return = Dataframe reconstruido a travez de predicciones y un diccionario que representa
        un measurement (accuracy para columnas categoricas, mae para columnas de tipo numero) que
        demuestra 'la performance' que obtuvimos de la red neuronal durante el training.
        """
        copy = df.copy()
        #Excluimos columnas indicadas
        for columna in columnas_excluir:
            copy = copy.drop(columna,1)

        goodness_of_fits= {}
        for col in copy:
            if not all(pd.notnull(copy[col].values)):
                copy[col],goodness_of_fits[col] = self.__predecir_no_definidos__(copy,col)

        return copy,goodness_of_fits



