from sympy import symbols, sympify, integrate, N

def calcular_integral_definida(expressao: str, variavel: str, a: float, b: float):
    """
    Calcula a integral definida de uma expressão simbólica.

    Args:
        expressao (str): A função a ser integrada, como string (ex: 'x**2 + 3*x').
        variavel (str): A variável de integração (ex: 'x').
        a (float): Limite inferior da integral.
        b (float): Limite superior da integral.

    Returns:
        dict: Contém a expressão simbólica da integral e o valor numérico.
    """
    x = symbols(variavel)
    func = sympify(expressao)
    integral_simbolica = integrate(func, (x, a, b))
    integral_numerica = N(integral_simbolica)

    return {
        "integral_simbolica": integral_simbolica,
        "integral_numerica": float(integral_numerica)
    }

# Exemplo de uso:
if __name__ == "__main__":
    resultado = calcular_integral_definida("x**2 + 3*x", "x", 0, 2)
    print("Resultado simbólico:", resultado["integral_simbolica"])
    print("Resultado numérico:", resultado["integral_numerica"])
