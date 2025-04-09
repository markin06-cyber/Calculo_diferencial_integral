import streamlit as st
from sympy import symbols, sympify, integrate, lambdify, latex
import matplotlib.pyplot as plt
import numpy as np

def mostrar_passos_integracao(func, variaveis, limites):
    resultado = func
    for var, (a, b) in reversed(list(zip(variaveis, limites))):
        st.latex(f"\\int_{{{a}}}^{{{b}}} {latex(resultado)} \\, d{var}")
        resultado = integrate(resultado, (var, a, b))
        st.latex(f"= {latex(resultado)}")
    return resultado

def integral_dupla(expr: str, var1: str, a1: float, b1: float,
                   var2: str, a2: float, b2: float,
                   plot: bool = False):
    x, y = symbols(f"{var1} {var2}")
    func = sympify(expr)

    if plot:
        try:
            fig, ax = plt.subplots()
            X = np.linspace(a1, b1, 300)
            ax.fill_between(X, a2, b2, color='skyblue', alpha=0.5)
            ax.hlines([a2, b2], a1, b1, colors='black', linestyles='--', linewidth=0.7)
            ax.vlines([a1, b1], a2, b2, colors='black', linestyles='--', linewidth=0.7)
            ax.set_xlabel(var1)
            ax.set_ylabel(var2)
            ax.set_title("Região de Integração Retangular")
            plt.grid(True)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Erro ao exibir gráfico: {e}")

    resultado = mostrar_passos_integracao(func, [x, y], [(a1, b1), (a2, b2)])
    return resultado

def integral_dupla_curvilinea(expr: str,
                               var_externa: str, a: float, b: float,
                               var_interna: str, g1: str, g2: str,
                               plot: bool = False):
    x, y = symbols(f"{var_externa} {var_interna}")
    func = sympify(expr)
    g1_expr = sympify(g1)
    g2_expr = sympify(g2)

    if plot:
        try:
            g1_func = lambdify(x, g1_expr, modules=["numpy"])
            g2_func = lambdify(x, g2_expr, modules=["numpy"])
            x_vals = np.linspace(a, b, 300)
            y1_vals = g1_func(x_vals)
            y2_vals = g2_func(x_vals)

            fig, ax = plt.subplots()
            ax.fill_between(x_vals, y1_vals, y2_vals, color='lightgreen', alpha=0.5, label='Região integrada')
            ax.plot(x_vals, y1_vals, 'b--', label=f'g1({var_externa})')
            ax.plot(x_vals, y2_vals, 'g--', label=f'g2({var_externa})')
            ax.set_xlabel(var_externa)
            ax.set_ylabel(var_interna)
            ax.set_title("Região de Integração Curvilínea")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Erro ao gerar gráfico: {e}")

    st.markdown("### Passo a passo da integração")
    intermedia = integrate(func, (y, g1_expr, g2_expr))
    st.latex(f"\\int_{{{latex(g1_expr)}}}^{{{latex(g2_expr)}}} {latex(func)} \\, d{var_interna} = {latex(intermedia)}")
    resultado = integrate(intermedia, (x, a, b))
    st.latex(f"\\int_{{{a}}}^{{{b}}} {latex(intermedia)} \\, d{var_externa} = {latex(resultado)}")

    return resultado

def integral_tripla(expr, var1, a1, b1, var2, a2, b2, var3, a3, b3, **kwargs):
    plot = kwargs.get("plot", False)

    x, y, z = symbols(f"{var1} {var2} {var3}")
    func = sympify(expr)

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        dx, dy, dz = b1 - a1, b2 - a2, b3 - a3
        ax.bar3d(a1, a2, a3, dx, dy, dz, alpha=0.3, color='cyan', edgecolor='black')
        ax.set_xlabel(var1)
        ax.set_ylabel(var2)
        ax.set_zlabel(var3)
        ax.set_title("Região da Integral Tripla (Paralelepípedo)")
        st.pyplot(fig)

    resultado = mostrar_passos_integracao(func, [x, y, z], [(a1, b1), (a2, b2), (a3, b3)])
    return resultado
