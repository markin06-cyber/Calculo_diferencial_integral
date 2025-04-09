from sympy import symbols, sympify, diff, sqrt, lambdify
import numpy as np
import plotly.graph_objects as go
import streamlit as st

def analisar_funcao_vetorial_com_grafico(expr_x: str, expr_y: str, expr_z: str, parametro: str):
    """
    Analisa uma função vetorial de uma variável real e plota seu gráfico 3D.

    Args:
        expr_x (str): Componente x(t) da função.
        expr_y (str): Componente y(t) da função.
        expr_z (str): Componente z(t) da função.
        parametro (str): Nome do parâmetro (ex: 't').

    Exibe:
        Resultados analíticos + gráfico 3D da curva vetorial.
    """
    t = symbols(parametro)
    x_t = sympify(expr_x)
    y_t = sympify(expr_y)
    z_t = sympify(expr_z)

    # Vetor posição e derivadas
    r = [x_t, y_t, z_t]
    r_prime = [diff(comp, t) for comp in r]
    r_prime_norm = sqrt(sum(comp**2 for comp in r_prime))
    r_double_prime = [diff(comp, t) for comp in r_prime]

    # Exibir resultados
    st.markdown("**Vetor posição:**")
    st.latex(r)
    st.markdown("**Velocidade:**")
    st.latex(r_prime)
    st.markdown("**Módulo da velocidade:**")
    st.latex(r_prime_norm)
    st.markdown("**Aceleração:**")
    st.latex(r_double_prime)

    # Gráfico da curva vetorial
    st.markdown("**Gráfico da curva vetorial**")

    t_vals = np.linspace(0, 2*np.pi, 300)

    fx = lambdify(t, r[0], modules=["numpy"])
    fy = lambdify(t, r[1], modules=["numpy"])
    fz = lambdify(t, r[2], modules=["numpy"])

    x_vals = fx(t_vals)
    y_vals = fy(t_vals)
    z_vals = fz(t_vals)

    fig = go.Figure(data=go.Scatter3d(
        x=x_vals, y=y_vals, z=z_vals,
        mode='lines',
        line=dict(color='blue', width=4)
    ))
    fig.update_layout(
        scene=dict(
            xaxis_title='x(t)',
            yaxis_title='y(t)',
            zaxis_title='z(t)'
        ),
        width=700,
        height=500,
        margin=dict(r=20, l=20, b=20, t=40)
    )
    st.plotly_chart(fig)

# Exemplo de uso
if __name__ == "__main__":
    st.set_page_config(page_title="Função Vetorial com Gráfico")
    st.title("Análise de Função Vetorial")
    expr_x = st.text_input("x(t):", "cos(t)")
    expr_y = st.text_input("y(t):", "sin(t)")
    expr_z = st.text_input("z(t):", "t")
    parametro = st.text_input("Parâmetro:", "t")

    if st.button("Analisar"):
        analisar_funcao_vetorial_com_grafico(expr_x, expr_y, expr_z, parametro)
