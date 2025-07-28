import marimo

__generated_with = "0.14.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    from chempy import electrolytes
    return (electrolytes,)


@app.cell
def _(electrolytes):
    print(electrolytes)
    return


@app.cell
def _(electrolytes):
    #https://pythonhosted.org/chempy/chempy.html?highlight=electrolytes#module-chempy.electrolytes
    #relative permittivity of solvent, float
    eps_r = 78.3 #water at 25C, find better source
    #Temperature (default: assume Kelvin), float with unit
    T = 303 #Temp the monitor says
    #density of the solvet (default: assume kg/m**3), float
    rho = 997/1000 #water
    #Reference molality, optionally with unit (amount / mass) IUPAC defines it as 1 mol/kg. (default: 1)., float
    b0 = 1

    A = electrolytes.A(eps_r=eps_r, T=T, rho=rho, b0=b0)

    print(A)
    return


@app.cell
def _(electrolytes):
    # from chempy import electrolytes.ionic_strength

    print(electrolytes.ionic_strength)
    return


if __name__ == "__main__":
    app.run()
