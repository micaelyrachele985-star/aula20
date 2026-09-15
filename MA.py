# ==========================================================
# DASHBOARD DE ANÁLISE DE DADOS DE ESPORTES
# ==========================================================

import pandas as pd
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.ensemble import RandomForestClassifier


# ==========================================================
# 1 - CARREGAR O ARQUIVO
# ==========================================================

dados = pd.read_csv("Players.csv")

print("========================================")
print("DADOS CARREGADOS")
print("========================================")
print(dados.head())

print("\nCOLUNAS DO ARQUIVO:")
print(dados.columns.tolist())


# ==========================================================
# 2 - LIMPAR OS NOMES DAS COLUNAS
# ==========================================================

dados.columns = dados.columns.str.strip()


# ==========================================================
# 3 - TRATAR VALORES FALTANTES
# ==========================================================

# Valores numéricos recebem a média da coluna
colunas_numericas = dados.select_dtypes(
    include="number"
).columns

for coluna in colunas_numericas:

    dados[coluna] = dados[coluna].fillna(
        dados[coluna].mean()
    )


# Valores de texto recebem "Desconhecido"
colunas_texto = dados.select_dtypes(
    include="object"
).columns

for coluna in colunas_texto:

    dados[coluna] = dados[coluna].fillna(
        "Desconhecido"
    )


# ==========================================================
# 4 - ENCONTRAR AS COLUNAS IMPORTANTES
# ==========================================================

def encontrar_coluna(lista):

    for nome in lista:

        if nome in dados.columns:
            return nome

    return None


coluna_jogador = encontrar_coluna([
    "Player",
    "player",
    "PLAYER"
])

coluna_altura = encontrar_coluna([
    "height",
    "Height",
    "HEIGHT"
])

coluna_peso = encontrar_coluna([
    "weight",
    "Weight",
    "WEIGHT"
])

coluna_nascimento = encontrar_coluna([
    "born",
    "Born",
    "BORN"
])

coluna_faculdade = encontrar_coluna([
    "college",
    "College",
    "collage",
    "Collage"
])

coluna_estado = encontrar_coluna([
    "birth_state",
    "Birth State",
    "birth state",
    "Birth_State"
])


# ==========================================================
# 5 - MOSTRAR AS COLUNAS ENCONTRADAS
# ==========================================================

print("\n========================================")
print("COLUNAS UTILIZADAS")
print("========================================")

print("Jogador:", coluna_jogador)
print("Altura:", coluna_altura)
print("Peso:", coluna_peso)
print("Nascimento:", coluna_nascimento)
print("Faculdade:", coluna_faculdade)
print("Estado:", coluna_estado)


# ==========================================================
# 6 - CRIAR NOVA VARIÁVEL
# CLASSIFICAÇÃO POR ALTURA
# ==========================================================

if coluna_altura is not None:

    dados["Classificacao_Altura"] = pd.cut(
        dados[coluna_altura],
        bins=[0, 180, 200, 250],
        labels=[
            "Baixo",
            "Médio",
            "Alto"
        ]
    )


# ==========================================================
# 7 - CODIFICAR VARIÁVEL CATEGÓRICA
# ==========================================================

if "Classificacao_Altura" in dados.columns:

    dados["Classificacao_Codigo"] = (
        dados["Classificacao_Altura"]
        .map({
            "Baixo": 0,
            "Médio": 1,
            "Alto": 2
        })
    )


# ==========================================================
# 8 - ANÁLISE ESTATÍSTICA
# ==========================================================

print("\n========================================")
print("ANÁLISE ESTATÍSTICA")
print("========================================")


# Altura
if coluna_altura is not None:

    media_altura = dados[coluna_altura].mean()
    mediana_altura = dados[coluna_altura].median()
    minimo_altura = dados[coluna_altura].min()
    maximo_altura = dados[coluna_altura].max()

    print("\nALTURA")
    print("Média:", round(media_altura, 2))
    print("Mediana:", round(mediana_altura, 2))
    print("Mínimo:", minimo_altura)
    print("Máximo:", maximo_altura)


# Peso
if coluna_peso is not None:

    media_peso = dados[coluna_peso].mean()
    mediana_peso = dados[coluna_peso].median()
    minimo_peso = dados[coluna_peso].min()
    maximo_peso = dados[coluna_peso].max()

    print("\nPESO")
    print("Média:", round(media_peso, 2))
    print("Mediana:", round(mediana_peso, 2))
    print("Mínimo:", minimo_peso)
    print("Máximo:", maximo_peso)


# Ano de nascimento
if coluna_nascimento is not None:

    media_nascimento = dados[coluna_nascimento].mean()
    mediana_nascimento = dados[coluna_nascimento].median()
    minimo_nascimento = dados[coluna_nascimento].min()
    maximo_nascimento = dados[coluna_nascimento].max()

    print("\nANO DE NASCIMENTO")
    print("Média:", round(media_nascimento, 2))
    print("Mediana:", round(mediana_nascimento, 2))
    print("Mínimo:", minimo_nascimento)
    print("Máximo:", maximo_nascimento)


# ==========================================================
# 9 - DESCRIÇÃO ESTATÍSTICA
# ==========================================================

print("\n========================================")
print("DESCRIÇÃO ESTATÍSTICA")
print("========================================")

print(
    dados.select_dtypes(
        include="number"
    ).describe()
)


# ==========================================================
# 10 - INSIGHTS
# ==========================================================

def gerar_insights():

    texto = ""

    if coluna_altura is not None:

        texto += (
            "• A altura média dos jogadores é "
            f"{media_altura:.2f} cm.\n\n"
        )

        texto += (
            "• A maior altura encontrada foi "
            f"{maximo_altura:.0f} cm.\n\n"
        )

        texto += (
            "• A menor altura encontrada foi "
            f"{minimo_altura:.0f} cm.\n\n"
        )


    if coluna_peso is not None:

        texto += (
            "• O peso médio dos jogadores é "
            f"{media_peso:.2f} kg.\n\n"
        )

        texto += (
            "• O maior peso encontrado foi "
            f"{maximo_peso:.0f} kg.\n\n"
        )


    if (
        coluna_altura is not None
        and coluna_peso is not None
    ):

        correlacao = dados[
            [coluna_altura, coluna_peso]
        ].corr().iloc[0, 1]

        texto += (
            "• A correlação entre altura e peso é "
            f"{correlacao:.2f}.\n\n"
        )


        if correlacao > 0:

            texto += (
                "• Existe uma relação positiva: "
                "jogadores mais altos tendem a "
                "apresentar maior peso."
            )

        else:

            texto += (
                "• A relação entre altura e peso "
                "é negativa ou muito fraca."
            )


    return texto


# ==========================================================
# 11 - MACHINE LEARNING
# ==========================================================

modelo = None

if (
    coluna_altura is not None
    and coluna_peso is not None
):

    # Criar classificação de porte
    # baseada na altura

    dados["Desempenho"] = (
        dados["Classificacao_Altura"]
        .map({
            "Baixo": 0,
            "Médio": 1,
            "Alto": 2
        })
    )

    X = dados[
        [coluna_altura, coluna_peso]
    ]

    y = dados["Desempenho"]


    modelo = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    modelo.fit(X, y)


# ==========================================================
# 12 - CRIAR A JANELA
# ==========================================================

janela = tk.Tk()

janela.title(
    "Dashboard de Análise de Esportes"
)

janela.geometry(
    "1100x750"
)


# ==========================================================
# 13 - TÍTULO
# ==========================================================

titulo = tk.Label(
    janela,
    text="🏀 DASHBOARD DE ANÁLISE DE ESPORTES",
    font=("Arial", 20, "bold")
)

titulo.pack(pady=10)


subtitulo = tk.Label(
    janela,
    text="Análise de jogadores e suas características",
    font=("Arial", 12)
)

subtitulo.pack()


# ==========================================================
# 14 - ÁREA DOS GRÁFICOS
# ==========================================================

figura, eixo = plt.subplots(
    figsize=(9, 5)
)

canvas = FigureCanvasTkAgg(
    figura,
    master=janela
)

canvas.get_tk_widget().pack(
    fill=tk.BOTH,
    expand=True
)


# ==========================================================
# 15 - GRÁFICO DE BARRAS
# COMPARAÇÃO DE ESTATÍSTICAS
# ==========================================================

def grafico_barras():

    eixo.clear()


    if coluna_altura is not None:

        # Média de altura por classificação

        medias = dados.groupby(
            "Classificacao_Altura",
            observed=True
        )[coluna_altura].mean()


        eixo.bar(
            medias.index,
            medias.values
        )


        eixo.set_title(
            "Comparação da Média de Altura"
        )

        eixo.set_xlabel(
            "Classificação dos Jogadores"
        )

        eixo.set_ylabel(
            "Altura Média (cm)"
        )


    figura.tight_layout()

    canvas.draw()


# ==========================================================
# 16 - GRÁFICO DE DISPERSÃO
# ALTURA X PESO
# ==========================================================

def grafico_dispersao():

    eixo.clear()


    if (
        coluna_altura is not None
        and coluna_peso is not None
    ):

        eixo.scatter(
            dados[coluna_altura],
            dados[coluna_peso]
        )


        eixo.set_title(
            "Relação entre Altura e Peso"
        )

        eixo.set_xlabel(
            "Altura (cm)"
        )

        eixo.set_ylabel(
            "Peso (kg)"
        )


    figura.tight_layout()

    canvas.draw()


# ==========================================================
# 17 - GRÁFICO DE PIZZA
# DISTRIBUIÇÃO DOS JOGADORES
# ==========================================================

def grafico_pizza():

    eixo.clear()


    if "Classificacao_Altura" in dados.columns:

        distribuicao = dados[
            "Classificacao_Altura"
        ].value_counts()


        eixo.pie(
            distribuicao.values,
            labels=distribuicao.index,
            autopct="%1.1f%%"
        )


        eixo.set_title(
            "Distribuição dos Jogadores por Altura"
        )


    canvas.draw()


# ==========================================================
# 18 - MOSTRAR ESTATÍSTICAS
# ==========================================================

def mostrar_estatisticas():

    texto = "📊 ANÁLISE ESTATÍSTICA\n\n"


    if coluna_altura is not None:

        texto += (
            "ALTURA\n"
            f"Média: {media_altura:.2f} cm\n"
            f"Mediana: {mediana_altura:.2f} cm\n"
            f"Mínimo: {minimo_altura:.0f} cm\n"
            f"Máximo: {maximo_altura:.0f} cm\n\n"
        )


    if coluna_peso is not None:

        texto += (
            "PESO\n"
            f"Média: {media_peso:.2f} kg\n"
            f"Mediana: {mediana_peso:.2f} kg\n"
            f"Mínimo: {minimo_peso:.0f} kg\n"
            f"Máximo: {maximo_peso:.0f} kg\n\n"
        )


    if coluna_nascimento is not None:

        texto += (
            "ANO DE NASCIMENTO\n"
            f"Média: {media_nascimento:.2f}\n"
            f"Mediana: {mediana_nascimento:.2f}\n"
            f"Mínimo: {minimo_nascimento:.0f}\n"
            f"Máximo: {maximo_nascimento:.0f}\n"
        )


    messagebox.showinfo(
        "Estatísticas",
        texto
    )


# ==========================================================
# 19 - MOSTRAR INSIGHTS
# ==========================================================

def mostrar_insights():

    texto = gerar_insights()

    messagebox.showinfo(
        "💡 Insights",
        texto
    )


# ==========================================================
# 20 - IMPORTÂNCIA DAS CARACTERÍSTICAS
# ==========================================================

def mostrar_importancia():

    if modelo is None:

        messagebox.showinfo(
            "Machine Learning",
            "O modelo não pôde ser criado."
        )

        return


    importancia_altura = (
        modelo.feature_importances_[0]
    )

    importancia_peso = (
        modelo.feature_importances_[1]
    )


    texto = (
        "🤖 IMPORTÂNCIA DAS CARACTERÍSTICAS\n\n"
        f"Altura: {importancia_altura:.2f}\n"
        f"Peso: {importancia_peso:.2f}\n\n"
        "O modelo Random Forest foi utilizado "
        "para analisar as características dos jogadores."
    )


    messagebox.showinfo(
        "Machine Learning",
        texto
    )


# ==========================================================
# 21 - INFORMAÇÃO SOBRE VITÓRIAS
# ==========================================================

def mostrar_vitorias():

    texto = (
        "🏆 PREVISÃO DE VITÓRIAS\n\n"
        "O arquivo Players.csv contém dados "
        "individuais dos jogadores.\n\n"
        "Ele não possui resultados de partidas "
        "ou uma coluna indicando vitória/derrota.\n\n"
        "Por isso, não é correto criar uma "
        "previsão de vitória inventando dados."
    )


    messagebox.showinfo(
        "Previsão de Vitórias",
        texto
    )


# ==========================================================
# 22 - BOTÕES
# ==========================================================

frame_botoes = tk.Frame(janela)

frame_botoes.pack(
    pady=10
)


botao_barras = tk.Button(
    frame_botoes,
    text="📊 Barras",
    command=grafico_barras,
    width=15
)

botao_barras.pack(
    side=tk.LEFT,
    padx=4
)


botao_dispersao = tk.Button(
    frame_botoes,
    text="📈 Dispersão",
    command=grafico_dispersao,
    width=15
)

botao_dispersao.pack(
    side=tk.LEFT,
    padx=4
)


botao_pizza = tk.Button(
    frame_botoes,
    text="🥧 Pizza",
    command=grafico_pizza,
    width=15
)

botao_pizza.pack(
    side=tk.LEFT,
    padx=4
)


botao_estatisticas = tk.Button(
    frame_botoes,
    text="📋 Estatísticas",
    command=mostrar_estatisticas,
    width=15
)

botao_estatisticas.pack(
    side=tk.LEFT,
    padx=4
)


botao_insights = tk.Button(
    frame_botoes,
    text="💡 Insights",
    command=mostrar_insights,
    width=15
)

botao_insights.pack(
    side=tk.LEFT,
    padx=4
)


botao_ml = tk.Button(
    frame_botoes,
    text="🤖 Machine Learning",
    command=mostrar_importancia,
    width=20
)

botao_ml.pack(
    side=tk.LEFT,
    padx=4
)


botao_vitoria = tk.Button(
    frame_botoes,
    text="🏆 Vitórias",
    command=mostrar_vitorias,
    width=15
)

botao_vitoria.pack(
    side=tk.LEFT,
    padx=4
)


# ==========================================================
# 23 - MOSTRAR GRÁFICO INICIAL
# ==========================================================

grafico_barras()


# ==========================================================
# 24 - INICIAR DASHBOARD
# ==========================================================

janela.mainloop()