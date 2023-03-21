'''
    <--Pandas-->
        O primeiro passo para qualquer projeto de Inteligência Artificial é a familiarização com a base dados. Então para esse processo, utilizamos
        a biblioteca Pandas.

        Essa biblioteca é a ferramenta primária para qualquer pessoa, uma vez que permite explorar e manipular os dados.

        (1) Para importar essa biblioteca começamos com :
            > pip install pandas        (instalando a biblioteca no VScode)

        (2) Com a biblioteca já instalada, vamos usar ela da seguinte maneira:]
            > import pandas as pd

        (3) A partir da importação iremos agora pegar nossa base de dados
            (3.1) = Salvando nossa base de dados de maneira mais fácil e de ráido acesso
                > file_path = '../input/melbourne-housing-snapshot/melb_data.csv' 

            (3.2) = Lendo nossa base de dados e armazenando em um dataframe
                > melbourne_data = pd.read_csv(melbourne_file_path)   
        
        (4) Trazendo as informações primárias da nossa base de dados
                > print(melbourne_data.describe())
            
            (4.1) Interpletando as informações
                Os resultados mostram 8 números para cada coluna em seu conjunto de dados original. 
                    . count: a contagem, mostra quantas linhas têm valores não omissos.
                    . mean : que é a média. 
                    . std  : é o desvio padrão, que mede a dispersão numérica dos valores.
                    . min  : traz o menor valor para determinada coluna
                    . max  : traz o maior valor para determinada coluna
                    . 25%  : é o número que é maior que 25% dos dados da coulna
                    : 50%  : é o número que é maior que 50% dos dados da coulna
                    . 75%  :é o número que é maior que 75% dos dados da coulna

            (4.2) Selecionando os dados para o modelo
                Para escolher variáveis/colunas, precisaremos ver uma lista de todas as colunas do conjunto de dados. Isso é feito com a propriedade de colunas do DataFrame 
                    > melbourne_data.columns            (vai trazer todas as colunas de dados)
                
                Para tirar os valores vazios de uma base de dados usamos:
                    > melbourne_data = melbourne_data.dropna(axis=0)          
                        .axis 1 = eixo das colunas
                        .axis 0 = eixo das linhas    
                
        (5) Selecionando o alvo de previsão
            Usaremos a notação de ponto para selecionar a coluna que queremos prever, que é chamada de destino da previsão. Por convenção, a meta de previsão é chamada de y. 
            Portanto, o código que precisamos para salvar os preços das casas nos dados de Melbourne é
                > y = melbourne_data.Price

        (6) Escolhendo "Recursos"
            As colunas inseridas em nosso modelo (e posteriormente usadas para fazer previsões) são chamadas de "recursos". No nosso caso, essas seriam as colunas usadas para 
            determinar o preço da casa. Às vezes, você usará todas as colunas, exceto o destino, como recursos. Outras vezes, você ficará melhor com menos recursos.
                > melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

            By convention, this data is called X.
                > X = melbourne_data[melbourne_features]
            
            Vamos revisar rapidamente os dados que usaremos para prever os preços das casas usando o método describe e o método head, que mostra as primeiras linhas.
                > X.describe()
                > X.head()

        (7) Criando o modelo
            . Defina: Que tipo de modelo será? Uma árvore de decisão? Algum outro tipo de modelo?

            . Ajuste: capture padrões dos dados fornecidos. Este é o coração da modelagem.

            . Prever: Exatamente o que parece

            . Avaliar: determine a precisão das previsões do modelo.

            (7.1) Criando
                > from sklearn.tree import DecisionTreeRegressor

                 (7.1.1) Definir modelo. Especifique um número para random_state para garantir os mesmos resultados a cada execução
                        > melbourne_model = DecisionTreeRegressor(random_state=1)

                 (7.1.2.) Adequando modelo
                        > melbourne_model.fit(X, y)


'''     