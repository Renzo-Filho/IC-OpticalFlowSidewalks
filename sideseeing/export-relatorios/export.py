import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from jinja2 import Environment, FileSystemLoader
from datetime import datetime

def gerar_relatorio():
    """
    Função principal que executa a análise e gera o relatório HTML.
    """
    print("Iniciando a geração do relatório...")

    # --- 1. CARREGAR E ANALISAR DADOS ---
    df = sns.load_dataset('tips')

    # a. Métricas Principais
    metricas = {
        "Valor Total Faturado": f"R$ {df['total_bill'].sum():.2f}",
        "Gorjeta Média": f"R$ {df['tip'].mean():.2f}",
        "Dia Mais Movimentado": df['day'].mode()[0]
    }

    # b. Gráfico Estático
    plt.figure(figsize=(10, 6))
    sns.barplot(x='day', y='total_bill', data=df, estimator=sum, ci=None, palette='viridis')
    plt.title('Faturamento Total por Dia da Semana', fontsize=16)
    plt.xlabel('Dia da Semana')
    plt.ylabel('Faturamento Total (R$)')
    grafico_path = 'grafico_faturamento_dia.png'
    plt.savefig(grafico_path, dpi=300)
    plt.close() # Fecha a figura para não exibir no console
    print(f"Gráfico estático salvo em: {grafico_path}")

    # c. Gráfico Interativo
    fig = px.scatter(
        df, x='total_bill', y='tip', color='smoker', facet_col='sex',
        size='size', title='Relação entre Valor da Conta e Gorjeta',
        labels={'total_bill': 'Valor Total da Conta (R$)', 'tip': 'Gorjeta (R$)'}
    )
    grafico_interativo_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    print("Gráfico interativo gerado.")

    # d. Tabela Resumo
    resumo_por_dia = df.groupby(['day', 'time']).agg(
        gasto_medio=('total_bill', 'mean'),
        gorjeta_media=('tip', 'mean')
    ).reset_index().round(2)
    tabela_resumo_html = resumo_por_dia.to_html(index=False, classes='table', border=0)
    print("Tabela resumo gerada.")

    # --- 2. RENDERIZAR O TEMPLATE JINJA2 ---
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template('template.html')

    contexto = {
        "titulo": "Relatório de Análise de Gorjetas",
        "data_geracao": datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
        "metricas": metricas,
        "grafico_faturamento_path": grafico_path,
        "grafico_interativo_html": grafico_interativo_html,
        "tabela_resumo_html": tabela_resumo_html
    }

    html_final = template.render(contexto)
    print("Template renderizado com sucesso.")

    # --- 3. SALVAR O RELATÓRIO FINAL ---
    with open('template-final.html', 'w', encoding='utf-8') as f:
        f.write(html_final)

    print("Relatório 'template-final.html' gerado com sucesso!")

if __name__ == '__main__':
    gerar_relatorio()