import pandas as pd
from Preprocesadora import Preprocesadora

#df = pd.read_csv("house_living_cost.csv")
dic = {'Cant A':[4,5,1,6,5],'Cant B':[15,36,20,26,None],'Cant C':[1.2,1,0,-1,-0.8],'Sexo':["F","H","F",None,None]
       ,'Cat Altura':["Alto","Bajo","Mediano","Bajo",None]}
df = pd.DataFrame(dic)

prep = Preprocesadora(df)
prep.remplazar_no_definidos()
print(prep.get_dataframe())
