import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from sympy import latex, sympify, symbols, integrate, lambdify, N
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from core.funcoes_vetoriais import analisar_funcao_vetorial_com_grafico
from core.varias_variaveis import calcular_limite, derivadas_parciais, calcular_gradiente
from core.integrais_definidas import calcular_integral_definida

# Funções de integrais múltiplas

st.set_page_config(
    page_title="Calculadora de Cálculo",
    layout="centered",
    page_icon="🧮"
)

st.title("🧮 Calculadora de Cálculo Diferencial e Integral")

def integral_dupla(expr, var1, a1, b1, var2, a2, b2, plot=False):
    x, y = symbols(f"{var1} {var2}")
    func = sympify(expr)

    passo1 = integrate(func, (y, a2, b2))
    passo2 = integrate(passo1, (x, a1, b1))

    passos = [
        f"\\int_{{{a1}}}^{{{b1}}} \\int_{{{a2}}}^{{{b2}}} {latex(func)} \\, d{var2} \\, d{var1}",
        f"= \\int_{{{a1}}}^{{{b1}}} {latex(passo1)} \\, d{var1}",
        f"= {latex(passo2)}"
    ]

    if plot:
        f_lamb = lambdify((x, y), func, modules=['numpy'])
        x_vals = np.linspace(a1, b1, 50)
        y_vals = np.linspace(a2, b2, 50)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = f_lamb(X, Y)

        fig = go.Figure(data=[
            go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.8),
            go.Surface(z=np.zeros_like(Z), x=X, y=Y, showscale=False, opacity=0.3, colorscale='Greys')
        ])

        fig.update_layout(
            title="Visualização Interativa da Integral Dupla",
            scene=dict(
                xaxis_title=var1,
                yaxis_title=var2,
                zaxis_title='f(x,y)'
            ),
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

    return passo2, passos

import plotly.graph_objects as go

def integral_dupla_curvilinea(expr, var_ext, a, b, var_int, g1, g2, plot=False):
    x, y = symbols(f"{var_ext} {var_int}")
    func = sympify(expr)
    g1_expr = sympify(g1)
    g2_expr = sympify(g2)

    passo1 = integrate(func, (y, g1_expr, g2_expr))
    passo2 = integrate(passo1, (x, a, b))

    passos = [
        f"\\int_{{{a}}}^{{{b}}} \\int_{{{latex(g1_expr)}}}^{{{latex(g2_expr)}}} {latex(func)} \\, d{var_int} \\, d{var_ext}",
        f"= \\int_{{{a}}}^{{{b}}} {latex(passo1)} \\, d{var_ext}",
        f"= {latex(passo2)}"
    ]

    if plot:
        g1_func = lambdify(x, g1_expr, modules=["numpy"])
        g2_func = lambdify(x, g2_expr, modules=["numpy"])
        x_vals = np.linspace(float(a), float(b), 300)
        y1_vals = g1_func(x_vals)
        y2_vals = g2_func(x_vals)

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=x_vals, y=y1_vals, mode='lines', name='g1(x)', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=x_vals, y=y2_vals, mode='lines', name='g2(x)', line=dict(dash='dot')))

        fig.add_trace(go.Scatter(
            x=np.concatenate([x_vals, x_vals[::-1]]),
            y=np.concatenate([y1_vals, y2_vals[::-1]]),
            fill='toself',
            fillcolor='rgba(0,255,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Região Curvilínea'
        ))

        fig.update_layout(
            title="Região Curvilínea",
            xaxis_title=var_ext,
            yaxis_title=var_int,
            showlegend=True
        )

        st.plotly_chart(fig)

    return passo2, passos


def integral_tripla(expr, var1, a1, b1, var2, a2, b2, var3, a3, b3, plot=False):
    x, y, z = symbols(f"{var1} {var2} {var3}")
    func = sympify(expr)

    a1, b1, a2, b2, a3, b3 = map(sympify, [a1, b1, a2, b2, a3, b3])

    passo1 = integrate(func, (z, a3, b3))
    passo2 = integrate(passo1, (y, a2, b2))
    passo3 = integrate(passo2, (x, a1, b1))

    passos = [
        f"\\iiint_{{\\substack{{{latex(a1)} \\leq {var1} \\leq {latex(b1)} \\\\ {latex(a2)} \\leq {var2} \\leq {latex(b2)} \\\\ {latex(a3)} \\leq {var3} \\leq {latex(b3)}}}}} {latex(func)} \\, d{var3} \\, d{var2} \\, d{var1}",
        f"= \\iint {latex(passo1)} \\, d{var2} \\, d{var1}",
        f"= \\int {latex(passo2)} \\, d{var1}",
        f"= {latex(passo3)}"
    ]

    if plot:
        try:
            x_vals = np.linspace(float(a1), float(b1), 15)
            y_vals = np.linspace(float(a2), float(b2), 15)
            z_vals = np.linspace(float(a3), float(b3), 15)
            X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals)

            func_lambdified = lambdify((x, y, z), func, modules=["numpy"])
            F = func_lambdified(X, Y, Z).flatten()

            fig = go.Figure(data=go.Scatter3d(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                mode='markers',
                marker=dict(
                    size=4,
                    color=F,
                    colorscale='Viridis',
                    colorbar=dict(title=f"f({var1},{var2},{var3})"),
                    opacity=0.7
                )
            ))

            fig.update_layout(
                title="Visualização Interativa da Função f(x, y, z)",
                scene=dict(
                    xaxis_title=var1,
                    yaxis_title=var2,
                    zaxis_title=var3,
                ),
                height=600
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.warning(f"Não foi possível gerar o gráfico 3D da função: {e}")

    return passo3, passos
    x, y, z = symbols(f"{var1} {var2} {var3}")
    func = sympify(expr)

    a1, b1, a2, b2, a3, b3 = map(sympify, [a1, b1, a2, b2, a3, b3])

    passo1 = integrate(func, (z, a3, b3))
    passo2 = integrate(passo1, (y, a2, b2))
    passo3 = integrate(passo2, (x, a1, b1))

    passos = [
        f"\\iiint_{{\\substack{{{latex(a1)} \\leq {var1} \\leq {latex(b1)} \\\\ {latex(a2)} \\leq {var2} \\leq {latex(b2)} \\\\ {latex(a3)} \\leq {var3} \\leq {latex(b3)}}}}} {latex(func)} \\, d{var3} \\, d{var2} \\, d{var1}",
        f"= \\iint {latex(passo1)} \\, d{var2} \\, d{var1}",
        f"= \\int {latex(passo2)} \\, d{var1}",
        f"= {latex(passo3)}"
    ]

    if plot:
        try:
            x_vals = np.linspace(float(a1), float(b1), 10)
            y_vals = np.linspace(float(a2), float(b2), 10)
            z_vals = np.linspace(float(a3), float(b3), 10)
            X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals)

            func_lamb = lambdify((x, y, z), func, modules=["numpy"])
            F = func_lamb(X, Y, Z).flatten()

            fig = go.Figure(data=go.Scatter3d(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                mode='markers',
                marker=dict(
                    size=3,
                    color=F,
                    colorscale='Plasma',
                    opacity=0.7,
                    colorbar=dict(title='f(x, y, z)')
                )
            ))

            fig.update_layout(
                scene=dict(
                    xaxis_title=var1,
                    yaxis_title=var2,
                    zaxis_title=var3
                ),
                title="Visualização 3D da Função Integrada",
                margin=dict(l=0, r=0, b=0, t=40)
            )

            st.plotly_chart(fig)
        except Exception as e:
            st.warning(f"Erro ao gerar visualização 3D: {e}")

    return passo3, passos







# --- MENU LATERAL ---
menu = st.sidebar.selectbox(
    "Selecione a operação",
    [
        "Integrais Definidas",
        "Funções Vetoriais",
        "Funções de Várias Variáveis",
        "Integrais Múltiplas"
    ]
)

# --- INTERFACE: INTEGRAIS DEFINIDAS ---
if menu == "Integrais Definidas":
    st.title("Cálculo de Integral Definida")

    with st.form("form_integral_definida"):
        expressao = st.text_input("Função f(x)", value="x**2 + 3*x")
        variavel = st.text_input("Variável de integração", value="x")
        a = st.number_input("Limite inferior (a)", value=0.0)
        b = st.number_input("Limite superior (b)", value=2.0)
        submitted = st.form_submit_button("Calcular")

        if submitted:
            try:
                resultado = calcular_integral_definida(expressao, variavel, a, b)
                st.latex(f"\\int_{{{a}}}^{{{b}}} {expressao.replace('**', '^')} \\, d{variavel} = {latex(resultado['integral_simbolica'])}")
                st.success(f"Valor numérico aproximado: {resultado['integral_numerica']}")
            except Exception as e:
                st.error(f"Erro: {e}")

# --- INTERFACE: FUNÇÕES VETORIAIS ---
elif menu == "Funções Vetoriais":
    st.title("Análise de Funções Vetoriais")
    with st.form("form_func_vetorial"):
        x_expr = st.text_input("Componente x(t)", value="cos(t)")
        y_expr = st.text_input("Componente y(t)", value="sin(t)")
        z_expr = st.text_input("Componente z(t)", value="t")
        parametro = st.text_input("Parâmetro", value="t")
        submitted = st.form_submit_button("Analisar")

        if submitted:
            try:
                r = analisar_funcao_vetorial_com_grafico(x_expr, y_expr, z_expr, parametro)
                st.markdown("**Vetor posição:**")
                st.latex(latex(r['vetor_posicao']))
                st.markdown("**Velocidade:**")
                st.latex(latex(r['velocidade']))
                st.markdown("**Módulo da velocidade:**")
                st.latex(latex(N(r['modulo_velocidade'])))
                st.markdown("**Aceleração:**")
                st.latex(latex(r['aceleracao']))
            except Exception as e:
                st.error(f"Erro: {e}")

# --- INTERFACE: VÁRIAS VARIÁVEIS ---
elif menu == "Funções de Várias Variáveis":
    st.title("Análise de Funções de Várias Variáveis")

    with st.form("form_varias_variaveis"):
        expr = st.text_input("Função f(x, y)", value="x**2 * y + sin(y)")
        variaveis = st.text_input("Variáveis (separadas por vírgula)", value="x, y")
        calcular = st.form_submit_button("Calcular")

        if calcular:
            try:
                var_list = [v.strip() for v in variaveis.split(",")]
                derivs = derivadas_parciais(expr, var_list)
                grad = calcular_gradiente(expr, var_list)

                st.markdown("**Derivadas Parciais:**")
                for var, d in derivs.items():
                    st.latex(f"\\frac{{\\partial f}}{{\\partial {var}}} = {latex(d)}")

                st.markdown("**Gradiente:**")
                st.latex(latex(grad))

                st.markdown("---")
                st.subheader("Cálculo de Limite (iterado)")
                x0 = st.number_input(f"Valor para {var_list[0]}", value=0.0)
                y0 = st.number_input(f"Valor para {var_list[1]}", value=0.0)
                limite = calcular_limite(expr, var_list[0], x0, var_list[1], y0)
                st.latex(f"\\lim_{{{var_list[0]} \\to {x0}}} \\lim_{{{var_list[1]} \\to {y0}}} f(x,y) = {latex(limite)}")

            except Exception as e:
                st.error(f"Erro: {e}")

# --- INTERFACE: INTEGRAIS MÚLTIPLAS ---
elif menu == "Integrais Múltiplas":
    st.title("Cálculo de Integrais Múltiplas")

    tipo = st.radio("Tipo de integral", ["Dupla (Retangular)", "Dupla (Curvilínea)", "Tripla"])

    with st.form("form_integrais_multiplas"):
        expr = st.text_input("Função a ser integrada", value="x*y")

        if tipo == "Dupla (Retangular)":
            var1 = st.text_input("Variável externa", value="x")
            var2 = st.text_input("Variável interna", value="y")
            a1 = st.number_input(f"{var1}: Limite inferior", value=0.0)
            b1 = st.number_input(f"{var1}: Limite superior", value=1.0)
            a2 = st.number_input(f"{var2}: Limite inferior", value=0.0)
            b2 = st.number_input(f"{var2}: Limite superior", value=2.0)

            mostrar_grafico = st.checkbox("Mostrar gráfico da região", value=True)
            submitted = st.form_submit_button("Calcular")

            if submitted:
                try:
                    resultado, passos = integral_dupla(expr, var1, a1, b1, var2, a2, b2, plot=mostrar_grafico)
                    st.markdown("### Passo a passo:")
                    for p in passos:
                        st.latex(p)

                except Exception as e:
                    st.error(f"Erro: {e}")

        elif tipo == "Dupla (Curvilínea)":
            var_ext = st.text_input("Variável externa", value="x")
            var_int = st.text_input("Variável interna", value="y")
            a = st.number_input(f"{var_ext}: Limite inferior", value=0.0)
            b = st.number_input(f"{var_ext}: Limite superior", value=1.0)
            g1 = st.text_input("Função inferior g1(x)", value="x**2")
            g2 = st.text_input("Função superior g2(x)", value="sqrt(x)")
            mostrar_grafico = st.checkbox("Mostrar gráfico da região", value=True)
            submitted = st.form_submit_button("Calcular")

            if submitted:
                try:
                    resultado, passos = integral_dupla_curvilinea(expr, var_ext, a, b, var_int, g1, g2, plot=mostrar_grafico)
                    st.markdown("### Passo a passo:")
                    for p in passos:
                        st.latex(p)


                except Exception as e:
                    st.error(f"Erro: {e}")

        elif tipo == "Tripla":
            var1 = st.text_input("Variável 1", value="x")
            var2 = st.text_input("Variável 2", value="y")
            var3 = st.text_input("Variável 3", value="z")
            a1 = st.number_input(f"{var1}: Limite inferior", value=0.0)
            b1 = st.number_input(f"{var1}: Limite superior", value=1.0)
            a2 = st.number_input(f"{var2}: Limite inferior", value=0.0)
            b2 = st.number_input(f"{var2}: Limite superior", value=1.0)
            a3 = st.number_input(f"{var3}: Limite inferior", value=0.0)
            b3 = st.number_input(f"{var3}: Limite superior", value=1.0)

            mostrar_grafico = st.checkbox("Mostrar gráfico da região", value=True)
            submitted = st.form_submit_button("Calcular")
            if submitted:
                try:
                    resultado, passos = integral_tripla(expr, var1, a1, b1, var2, a2, b2, var3, a3, b3, plot=mostrar_grafico)
                    st.markdown("### Passo a passo:")
                    for p in passos:
                        st.latex(p)


                except Exception as e:
                    st.error(f"Erro: {e}")
