from sympy import symbols, sympify, limit, diff, Matrix

def calcular_limite(expr: str, var1: str, ponto1: float, var2: str, ponto2: float):
    """
    Calcula o limite de uma função de duas variáveis reais.

    Args:
        expr (str): Expressão da função (ex: '(x**2 * y) / (x**2 + y**2)').
        var1 (str): Primeira variável (ex: 'x').
        ponto1 (float): Valor para o qual var1 tende.
        var2 (str): Segunda variável (ex: 'y').
        ponto2 (float): Valor para o qual var2 tende.

    Returns:
        sympy.Expr: Limite da função.
    """
    x, y = symbols(var1 + " " + var2)
    func = sympify(expr)
    limite = limit(limit(func, x, ponto1), y, ponto2)  # Limite iterado
    return limite

def derivadas_parciais(expr: str, variaveis: list[str]):
    """
    Calcula as derivadas parciais de uma função em relação às variáveis fornecidas.

    Args:
        expr (str): Expressão da função.
        variaveis (list[str]): Lista de variáveis (ex: ['x', 'y']).

    Returns:
        dict: Derivadas parciais em relação a cada variável.
    """
    vars_sym = symbols(" ".join(variaveis))
    func = sympify(expr)
    derivadas = {str(var): diff(func, var) for var in vars_sym}
    return derivadas

def calcular_gradiente(expr: str, variaveis: list[str]):
    """
    Calcula o vetor gradiente de uma função de várias variáveis.

    Args:
        expr (str): Expressão da função.
        variaveis (list[str]): Lista de variáveis.

    Returns:
        Matrix: Vetor gradiente.
    """
    vars_sym = symbols(" ".join(variaveis))
    func = sympify(expr)
    grad = Matrix([diff(func, var) for var in vars_sym])
    return grad

# Exemplo de uso:
if __name__ == "__main__":
    f = "(x**2 * y) / (x**2 + y**2)"
    print("Limite:", calcular_limite(f, "x", 0, "y", 0))

    derivs = derivadas_parciais("x**2 * y + sin(y)", ["x", "y"])
    print("Derivadas parciais:", derivs)

    grad = calcular_gradiente("x**2 * y + sin(y)", ["x", "y"])
    print("Gradiente:", grad)
