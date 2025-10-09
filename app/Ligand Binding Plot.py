import marimo

__generated_with = "0.15.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    from cycler import cycler

    colors = [
        'white',
        '#DE8CDE',  # lilac
        '#00C2A8',  # teal
        '#FFB84D',  # warm amber
        '#57ABFF',  # bright blue
        '#FF8A8A',  # coral red
        '#8CE99A',  # mint green
        # '#A9A9AD',  # light grey
    ]

    linestyles = [
        '-',
        '--',
        ':',
        '-.',
        (0, (5, 1)),
        (0, (3, 5, 1, 5)),
        (0, (1, 1)),
    ]

    # Colors
    fig_bg = '#1B1B1D'    # figure background
    ax_bg = fig_bg   # axes background

    plt.rcParams['figure.facecolor'] = fig_bg
    plt.rcParams['axes.facecolor'] = ax_bg
    plt.rcParams['axes.edgecolor'] = '#333333'  # axes border
    plt.rcParams['axes.labelcolor'] = colors[0]
    plt.rcParams['xtick.color'] = colors[0]
    plt.rcParams['ytick.color'] = colors[0]
    plt.rcParams['text.color'] = colors[0]

    # plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

    plt.rcParams['axes.prop_cycle'] = cycler(color=colors) + cycler(
        linestyle=linestyles
    )
    return colors, plt


@app.cell
def _(colors, plt):
    import numpy as np
    from scipy.optimize import curve_fit

    def hill_equation(concentration, bmax, kd):
        """
        Hill equation for ligand binding

        Parameters:
        -----------
        concentration : array-like
            Ligand concentration (e.g., in M)
        bmax : float
            Maximum binding (typically ~100%)
        kd : float
            Dissociation constant (concentration at 50% binding)
        n : float
            Hill coefficient (cooperativity parameter)

        Returns:
        --------
        percent_bound : array-like
            Percentage of protein bound
        """
        return bmax * (concentration) / (kd + concentration)

    def fit_binding_curve(concentrations, percent_bound, initial_guess=None):
        """
        Fit experimental binding data to the Hill equation

        Parameters:
        -----------
        concentrations : array-like
            Ligand concentrations
        percent_bound : array-like
            Percentage of protein bound at each concentration
        initial_guess : tuple, optional
            Initial guess for (bmax, kd, n). Default: (100, median_conc, 1)

        Returns:
        --------
        params : tuple
            Fitted parameters (bmax, kd, n)
        pcov : array
            Covariance matrix
        """
        if initial_guess is None:
            median_conc = np.median(concentrations[concentrations > 0])
            initial_guess = (100, median_conc)

        # Fit the data
        params, pcov = curve_fit(
            hill_equation,
            concentrations,
            percent_bound,
            p0=initial_guess,
            bounds=([0, 0], [150, np.inf]),
            maxfev=10000,
        )

        return params, pcov

    def plot_binding_curve(
        concentrations, percent_bound, params=None, show_equation=True
    ):
        """
        Plot the binding curve with experimental data and fitted curve

        Parameters:
        -----------
        concentrations : array-like
            Experimental ligand concentrations
        percent_bound : array-like
            Experimental percentage bound
        params : tuple, optional
            Fitted parameters (bmax, kd, n)
        show_equation : bool
            Whether to display the equation on the plot
        """
        plt.figure(figsize=(10, 6))

        # Plot experimental data
        plt.scatter(
            concentrations,
            percent_bound,
            s=100,
            label='Experimental Data',
            zorder=5,
            linewidths=1.5,
        )

        # Plot fitted curve if parameters provided
        if params is not None:
            bmax, kd = params

            # Generate smooth curve
            conc_range = np.linspace(0, max(concentrations) * 1.2, 200)
            fitted_values = hill_equation(conc_range, bmax, kd)

            plt.plot(
                conc_range,
                fitted_values,
                'b-',
                linewidth=2.5,
                label='Fitted Curve',
                zorder=3,
                color=colors[1]
            )

            # Add horizontal line at 50% binding
            plt.axhline(
                y=50, linestyle='--', alpha=0.5, linewidth=1
            )
            plt.axvline(
                x=kd, linestyle='--', alpha=0.5, linewidth=1
            )

            if show_equation:
                equation_text = (
                    f'$K_d$ = {kd:.3f} M\n$B_{{max}}$ = {bmax:.2f}%\n'
                )
                # plt.text(
                #     0.05,
                #     0.95,
                #     equation_text,
                #     transform=plt.gca().transAxes,
                #     fontsize=11,
                #     verticalalignment='top',
                #     bbox=dict(boxstyle='round', alpha=0.1),
                # )

        plt.xlabel('Ligand Concentration (M)', fontsize=12, fontweight='bold')
        plt.ylabel('% Bound', fontsize=12, fontweight='bold')
        plt.title('Ligand Binding Curve', fontsize=14, fontweight='bold')
        plt.ylim(-5, 105)
        plt.grid(True, alpha=0.3, linestyle=':')
        plt.legend(fontsize=10, loc='lower right')
        plt.tight_layout()
        plt.savefig(f'figs/binding-{concentrations[-1]}.svg')
        plt.show()

    def analyze_binding_data(concentrations, percent_bound):
        """
        Complete analysis of binding data: fit curve and display results

        Parameters:
        -----------
        concentrations : array-like
            Ligand concentrations
        percent_bound : array-like
            Percentage bound at each concentration
        """
        # Convert to numpy arrays
        concentrations = np.array(concentrations, dtype=float)
        percent_bound = np.array(percent_bound, dtype=float)

        # Fit the curve
        params, pcov = fit_binding_curve(concentrations, percent_bound)
        bmax, kd = params

        # Calculate standard errors
        perr = np.sqrt(np.diag(pcov))

        # Calculate R-squared
        residuals = percent_bound - hill_equation(concentrations, *params)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((percent_bound - np.mean(percent_bound)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        # Print results
        print('=' * 60)
        print('LIGAND BINDING CURVE ANALYSIS RESULTS')
        print('=' * 60)
        print(f'\nFitted Parameters:')
        print(f'  Kd (Dissociation Constant): {kd:.4f} ± {perr[1]:.4f} M')
        print(f'  Bmax (Maximum Binding):     {bmax:.3f} ± {perr[0]:.3f} %')
        print(f'\nGoodness of Fit:')
        print(f'  R² = {r_squared:.4f}')
        print(f'  RMSE = {np.sqrt(ss_res/len(concentrations)):.3f}')

        print('\n' + '=' * 60)

        # Plot the results
        plot_binding_curve(concentrations, percent_bound, params)

        return params, pcov

    def analyze_data(element_data):
        concentrations = [float(row[0]) for row in element_data]
        shifts = [row[1] for row in element_data]
        min_shift = shifts[0]
    
        shift_changes = [shift - min_shift for shift in shifts]
        print(f'Range of shift changes for {element_data[0][2]}: {np.min(shift_changes)} to {np.max(shift_changes)}')

        max_shift_change = max(shift_changes) + 0.05

        shift_percentages = [
            (shift_change / max_shift_change) * 100
            for shift_change in shift_changes
        ]
    
        # Assuming analyze_binding_data is defined somewhere
        params, pcov = analyze_binding_data(concentrations, shift_percentages)
    
        return params, pcov

    # Example usage
    if __name__ == '__main__':
        magnesium = [
            [
                0.0,
                2.557808104376063,
                2.5329411043760635,
                2.4663431043760635,
                2.4414671043760627,
            ],
            [
                0.00016568950133257033,
                2.5697641666413245,
                2.5444251666413242,
                2.4678961666413244,
                2.4425081666413244,
            ],
            [
                0.00032233380934806644,
                2.5845465254825717,
                2.5588055254825717,
                2.4736975254825717,
                2.4478855254825715,
            ],
            [
                0.00048603607478041663,
                2.58998800514798,
                2.56398300514798,
                2.47050600514798,
                2.44429100514798,
            ],
            [
                0.0006643641506141569,
                2.6012765837973943,
                2.574670583797394,
                2.4713895837973943,
                2.4446795837973943,
            ],
            [
                0.0008129719211462715,
                2.6090159343132258,
                2.5820249343132256,
                2.4719149343132254,
                2.4448409343132256,
            ],
            [
                0.000990913447565343,
                2.6158515837973946,
                2.5886005837973944,
                2.4730385837973943,
                2.4457095837973943,
            ],
            [
                0.0011444642083250345,
                2.615115077769385,
                2.5876730777693853,
                2.4684050777693853,
                2.4409010777693854,
            ],
            [
                0.001319198491716594,
                2.6235281377392212,
                2.595943137739221,
                2.473775137739221,
                2.446112137739221,
            ],
            [
                0.0015246711899502647,
                2.626723220828511,
                2.5990142208285114,
                2.4743892208285114,
                2.4466112208285113,
            ],
            [
                0.0017470439228487332,
                2.6280017286431714,
                2.600247728643171,
                2.4744837286431713,
                2.446666728643171,
            ],
            [
                0.001738163880435133,
                2.628709859447762,
                2.600928859447762,
                2.474444859447762,
                2.446607859447762,
            ],
        ]
        calcium = [
            [
                np.float64(1.3069759771669533e-05),
                2.557808104376063,
                2.5329411043760635,
                2.4663431043760635,
                2.4414671043760627,
            ],
            [
                np.float64(0.0004388693143677467),
                2.5697641666413245,
                2.5444251666413242,
                2.4678961666413244,
                2.4425081666413244,
            ],
            [
                np.float64(0.0008710045777422681),
                2.5845465254825717,
                2.5588055254825717,
                2.4736975254825717,
                2.4478855254825715,
            ],
            [
                np.float64(0.0012586831839796354),
                2.58998800514798,
                2.56398300514798,
                2.47050600514798,
                2.44429100514798,
            ],
            [
                np.float64(0.0017739854331590707),
                2.6012765837973943,
                2.574670583797394,
                2.4713895837973943,
                2.4446795837973943,
            ],
            [
                np.float64(0.0021617898365660893),
                2.6090159343132258,
                2.5820249343132256,
                2.4719149343132254,
                2.4448409343132256,
            ],
            [
                np.float64(0.002623578466452271),
                2.6158515837973946,
                2.5886005837973944,
                2.4730385837973943,
                2.4457095837973943,
            ],
            [
                np.float64(0.0029891191582014694),
                2.615115077769385,
                2.5876730777693853,
                2.4684050777693853,
                2.4409010777693854,
            ],
            [
                np.float64(0.003515920468134446),
                2.6235281377392212,
                2.595943137739221,
                2.473775137739221,
                2.446112137739221,
            ],
            [
                np.float64(0.003952538042608662),
                2.626723220828511,
                2.5990142208285114,
                2.4743892208285114,
                2.4466112208285113,
            ],
            [
                np.float64(0.004370864759411153),
                2.6280017286431714,
                2.600247728643171,
                2.4744837286431713,
                2.446666728643171,
            ],
            [
                np.float64(0.005000942855609513),
                2.628709859447762,
                2.600928859447762,
                2.474444859447762,
                2.446607859447762,
            ],
        ]
        # Example experimental data
        concentrations = np.array(
            [
                0.0,
                0.00016568950133257033,
                0.00032233380934806644,
                0.00048603607478041663,
                0.0006643641506141569,
                0.0008129719211462715,
                0.000990913447565343,
                0.0011444642083250345,
                0.001319198491716594,
                0.0015246711899502647,
                0.0017470439228487332,
                0.001738163880435133,
            ]
        )  # M
        percent_bound = np.array(
            [
                0.0,
                14.77849554025797,
                33.050483370616405,
                39.776517509907976,
                53.72995849448198,
                63.29631525518813,
                71.74563687758153,
                70.83526598716409,
                81.23437285744168,
                85.1837100336974,
                86.76403151563204,
                87.6393287251463,
            ]
        )  # %

        magnesium_results = analyze_data(magnesium)
        calcium_results = analyze_data(calcium)

        # You can also use the functions individually:
        # params, pcov = fit_binding_curve(concentrations, percent_bound)
        # plot_binding_curve(concentrations, percent_bound, params)
    return


if __name__ == "__main__":
    app.run()
