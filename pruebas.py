import pandas as pd
from Preprocesadora import Preprocesadora

df = pd.read_csv("analcatdata_marketing.csv")
"""
dic = {'Cant A':[4,5,1,6,5],'Cant B':[15,36,20,26,None],'Cant C':[1.2,1,0,-1,-0.8],'Sexo':["F","H","F",None,None]
       ,'Cat Altura':["Alto","Bajo","Mediano","Bajo",None],'Edad':[15,16,18,None,None]}
df = pd.DataFrame(dic)
"""

prep = Preprocesadora()
reconstruido, perfomance = prep.reconstruir_por_prediccion(df,columnas_excluir=["X1a"])
#df = prep.obtener_filas_no_definidas_en_columna(df,"X1a")
print(df)

