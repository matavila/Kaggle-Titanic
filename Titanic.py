'''
    Passo a passo de um Projeto de Ciência de Dados
    (1) Entendimento do desafio 
    (2) Extração e obtenção dos dados
    (3) Ajuste de dados (Tratamento/Limpeza)
    (4) Análise Exploratória
    (5) Modelagem + Algorítmos (IA se necessário)
    (6) Interpretação dos resultados
'''
#<-----------------------Passo (1)------------------------>
'''
    Criar uma IA para fazer a previsão para seguinte pergunta:
        "Quais são os tipos de pessoas mais propensas a sobreviver ao desastre do titanic ? "

    Na coluna Survived:
        0 Sobreviveu
        1 Morreu
    Coluna sibspn (se tem ou não parentes com eles) e Parch(se tem filho ou pais com eles)

'''
#<-----------------------Passo (2)------------------------>
#Importações Gerais
import pandas as pd
import seaborn as sns                     #biblioteca de gráfico
import matplotlib.pyplot as plt           #biblioteca de gráfico

#Importação da IA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

#Obtenção dos dados
trainfile_pathway = r'C:\Users\Admin\OneDrive\Área de Trabalho\ESTUDOS\DEV-DIO\Python\YT_Exemple\Data Science Tasks\Steel\train.csv'             #Colocamos r".." para que seja lido o caminho como está escrito
testfile_pathway = r'C:\Users\Admin\OneDrive\Área de Trabalho\ESTUDOS\DEV-DIO\Python\YT_Exemple\Data Science Tasks\Steel\test.csv'
submissionfile_pathway = r'C:\Users\Admin\OneDrive\Área de Trabalho\ESTUDOS\DEV-DIO\Python\YT_Exemple\Data Science Tasks\Steel\gender_submission.csv'

#Extração
Tabela = pd.read_csv(trainfile_pathway)
print(Tabela.head(5))
Tabela_Teste = pd.read_csv(testfile_pathway)



#<-----------------------Passo (3)------------------------>
#print(Tabela.info())

'''
    Visualizando as informações percebemos que há 5 colunas com dados tipos string, 
        . Name
        . Sex
        . Ticket
        . Cabin
        . Embarked

    Onde, de primeira análise consigo verificar alguns dados desnecessários
'''

#Deletando a colunas desnecessária:
def Limpeza(Tabela):
    Tabela = Tabela.drop(["Ticket","Cabin","Name"],axis=1)

    #Tirando os valores vazios
    Tabela= Tabela.dropna()

    #Substituindo genero por número => 1 homem e 0 Mulher
    Tabela['Sex'].replace({'male': 0, 'female': 1}, inplace=True)

    #Substituindo o local de embarque para C = 0, Q=1, S=2 e U=3
    Tabela['Embarked'].replace({'C': 0, 'Q': 1, 'S':2 ,'U':3 }, inplace=True)
    return Tabela
New_Tabela = Limpeza(Tabela)
New_Test = Limpeza(Tabela_Teste)
print(New_Tabela.head(5))
print(New_Tabela.info())

#Agora temos toda a nossa base de dados convertidade em numeros

#<-----------------------Passo (4)------------------------>
#Iremos agora procurar correlações com os dados presentes na tabela
correlacao = New_Tabela.corr()
print(correlacao)

#Para melhorar a visualização iremos criar um gráfico
#sns.heatmap(correlacao, cmap="Blues", annot = True)                   #Cria o gráfico
#plt.show()                                                            #Exibe o gráfico pelo python e podemos exportar

'''
    Percebemos correlação forte:
        Pclass: - 0.35 (logo pior a classe (3), menor é a chance de sobreviver)
        Sex   : 0.56   (logo mulheres apresentam maior taxa de sobrevivência)
        Fare  : 0,26   (quanto maior a tarifa paga pelo passageiro, maior a taxa de sobrevivencia)
'''


#<-----------------------Passo (5)------------------------>
# Objetivo: Dado as características do passageiro, a nossa IA consiga prever o se o mesmo sobrevive
# Quer prever quem: Sobrevivencia (Y)

''' 
    Começaremos nossa etapa, separando a nossa base de dados em :
        . Dados Y = Survived
        . Dados X = Características (Pclass, sex, Fare)

    axis 1 = eixo das colunas
    axis 0 = eixo das linhas
'''
y = New_Tabela["Survived"]

x = New_Tabela.drop("Survived", axis=1)

#<-----------------------Passo (5.1)------------------------>

#---> Separando os dados de treino e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x,y, test_size=0.3,random_state = 1 ) 

#Criando a IA   (Nesse momento ela é burra)
modelo_regressaolinear = LinearRegression()
modelo_arvoredecisao = RandomForestRegressor()
modelo_logistica = LogisticRegression()

#Treinando a IA
modelo_regressaolinear.fit(x_treino,y_treino)
modelo_logistica.fit(x_treino,y_treino)
modelo_arvoredecisao.fit(x_treino,y_treino)

#Escolhendo o melhor modelo -> usaremos o R^2 para calcular o erro da variação (ou seja percentual de precisão)

previsao_regressãoLinear = modelo_regressaolinear.predict(x_teste)
previsao_ArvoreDeDecisao = modelo_arvoredecisao.predict(x_teste)
previsao_Logistica = modelo_logistica.predict(x_teste)

#Comparando R2
print(r2_score(y_teste,previsao_regressãoLinear))
print(r2_score(y_teste,previsao_ArvoreDeDecisao))
print(r2_score(y_teste,previsao_Logistica))

#Criando um arquivo para colocar as previsões
tabela_auxiliar = pd.DataFrame()
tabela_auxiliar["ID"] = New_Test["PassengerId"]

#Fazendo a previsão dos sobreviventes a partir da tabela teste
previsao = modelo_regressaolinear.predict(New_Test)

#Adicionando os resultados em uma tabela alternativa
tabela_auxiliar["y_teste"] = previsao
tabela_auxiliar["y_teste"] = round(tabela_auxiliar["y_teste"],0)
print(tabela_auxiliar)