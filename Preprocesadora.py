#Bibliotecas estructuras de datos
import pandas as pd
import numpy as np

#Clases utiles
from sklearn import preprocessing
from collections import Counter

#Funciones utiles
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

class Preprocesadora:
    """
    Clase que tiene el objetivo de preprocesar un dataframe con el objetivo que el mismo
    pueda ser usado como input para una red neuronal.
    """
    def __init__(self,df):
        self.df = df.copy()
        self.original = df.copy()

    def aplicar_one_hot_encoding(self):
        """
        Dado un dataframe, performa un OneHotEncoding sobre las columnas categoricas del dataframe y devu
        y guarda los resultados.
        :param df: Pandas dataframe
        """
        self.df = pd.get_dummies(self.df,dummy_na=True)

    def __is_dummy__(self,col_name):
        """
        La idea es la siguiente, si chekeamos 30 valores que sean distintos a Nan en una columna
        y los 30 son valores que son iguales a 0 o 1 podemos asumir que es una dummy variable. De
        esta manera nos ahorramos recorrer todas las filas

        :param col_name: Columna donde hacemos el chekeo
        :return: True si es una dummy variable(solo unos y ceros) False en caso contrario
        """
        long_dataframe = len(self.df.index)
        longitud_chekeo = long_dataframe if long_dataframe < 30 else 30
        i = 0
        if(not is_numeric_dtype(self.df[col_name])):
            return False
        while i < longitud_chekeo:
            if (np.isnan(i)):
                longitud_chekeo += 1 if (longitud_chekeo+1) < long_dataframe else 0
            if(self.df.at[i,col_name] != 0 and self.df.at[i,col_name] != 1):
                return False
            i += 1
        return True

    def aplicar_input_scaling(self):
        """
        Escala el input para que los mismos se encuentren en una misma escala y se
        facilite la convergencia/disminusca la varianza.
        :param df: Dataframe a escalar
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

        skewness_test = self.df.skew(0)
        self.dic_inv_transformations = {}
        for col in skewness_test.index:
            is_dummy_variable = self.__is_dummy__(col)
            if(not is_dummy_variable):
                if(skewness_test[col] <= 0.5 and skewness_test[col] >= -0.5):
                    self.df[col], inverse_transf_pointer = estandarizar_valores(self.df,col)
                else:
                    self.df[col], inverse_transf_pointer = normalizar_valores(self.df, col)
                self.dic_inv_transformations[col] = inverse_transf_pointer

    def deshacer_input_scaling(self):
        """
        Deshace la transformacion de input scaling aplicada al dataframe
        """
        for key,inv_trans in self.dic_inv_transformations.items():
            self.df[key] = inv_trans(self.df[key].values.reshape(-1,1))

    def remplazar_numericos_no_definidos(self,col,tipo_remplazo="mean"):
        """
        Dada una columna y un tipo de remplazo, remplaza los valores no definidos por el tipo
        de remplazo indicado, siendo los tipos posibles el remplazo por mean o median.
        :param col: Nombre de la columna en el dataframe
        :param tipo_remplazo: Tipo de remplazo a realizar 'mean'/'median'
        """
        if is_numeric_dtype(self.df[col]):
            if tipo_remplazo == "mean":
                self.df[col].fillna(self.df[col].mean(),inplace=True)
            elif tipo_remplazo == "median":
                self.df[col].fillna(self.df[col].median(), inplace=True)

    def remplazar_categoricos_no_definidos(self, col,most_frecuent=True):
        """
        Dado una columna y un tipo de remplazo, remplaza los valores no definidos por ese
        tipo de remplazoo, siendo los dos tipos posibles por el valor mas frecuente, o directamente
        una nueva categoria para los valores no definidos.
        :param col: Nombre de la columna
        :param most_frecuent: True en caso que se quiera remplazar los valores indefinidos por la categoria
        mas frecuente, False en caso que se cree una nueva categoria con el nombre "Undefined".
        """
        if is_string_dtype(self.df[col]):
            if most_frecuent:
                contador = Counter(self.df[col])
                mas_frecuente = contador.most_common(1)[0][0]
                self.df[col].fillna(mas_frecuente,inplace=True)
            else:
                self.df[col].fillna("Undefined", inplace=True)

    def remplazar_no_definidos(self):
        """
        La idea de esta funcion es de remplazar de manera rapida todos los valores no definidos
        en el dataframe, de manera tal que si la columna es de tipo numerico se remplaza por la mean,
        y si la misma es de tipo categorico se remplaza por la categoria mas frecuente.
        En caso de que se nececite ser mas especifico usar las otras funciones.
        """
        for col in self.df:
            if is_numeric_dtype(self.df[col]):
                self.remplazar_numericos_no_definidos(col)
            else:
                self.remplazar_categoricos_no_definidos(col)

    def get_dataframe(self):
        """
        Retorna el estado actual del dataframe
        :return: pandas dataframe
        """
        return self.df


