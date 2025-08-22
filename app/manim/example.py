from manim import *


def create_graph(x, y, xlabel=None, ylabel=None, scale_factor=0.8):
    pts = zip(x, y)
    axes = Axes(
        x_range=[min(x), max(x), (max(x) - min(x)) / 10],
        y_range=[min(y), max(y), (max(y) - min(y)) / 10],
        tips=False,
        axis_config={
            'include_numbers': True,
            'decimal_number_config': {'num_decimal_places': 2},
        },
    ).to_edge(DOWN)

    coord_points = [axes.coords_to_point(x, y) for x, y in pts]

    poly = VMobject()
    poly.set_points_as_corners(coord_points)
    poly.set_stroke('#521671', 3)

    dots = VGroup(*[Dot(p, radius=0.06, color=YELLOW) for p in coord_points])

    labels = VGroup(
        *[
            MathTex(f'({x},{y})').scale(0.5).next_to(pt, UR, buff=0.06)
            for (x, y), pt in zip(pts, coord_points)
        ]
    )

    # Adding axis labels
    if xlabel is not None:
        x_label = (
            MathTex(xlabel)
            .scale(0.7)
            .next_to(axes.x_axis.get_center(), RIGHT, buff=0.2)
        )

    if ylabel is not None:
        y_label = (
            MathTex(ylabel)
            .scale(0.7)
            .next_to(axes.y_axis.get_center(), UP, buff=0.2)
        )

    # Scale the axes and graph elements
    axes.scale(scale_factor)
    poly.scale(scale_factor)
    dots.scale(scale_factor)
    labels.scale(scale_factor)

    if xlabel is not None:
        x_label.scale(scale_factor)

    if ylabel is not None:
        y_label.scale(scale_factor)

    return poly, axes, dots, labels, x_label, y_label


class PlotPoints(Scene):
    def construct(self):
        self.camera.background_color = '#1e1e1e'

        phs = [
            2.65,
            3.04,
            2.99,
            3.31,
            3.33,
            2.96,
            3.83,
            3.65,
            4.36,
            4.61,
            4.7,
            4.9,
            5.13,
            5.32,
            5.48,
            5.59,
            5.79,
            6.0,
            6.8,
            6.3,
            6.51,
            6.55,
            7.09,
            7.53,
        ]

        citrate_couplings = [
            0.02620699999999987,
            0.026074500000000222,
            0.02601449999999983,
            0.02594950000000007,
            0.025904499999999997,
            0.0261015,
            0.02575700000000003,
            0.025643000000000082,
            0.025580500000000228,
            0.0254915,
            0.02543150000000005,
            0.02535200000000004,
            0.025285500000000072,
            0.025217000000000045,
            0.0251595,
            0.025150999999999923,
            0.025144500000000125,
            0.02511449999999993,
            0.025070499999999774,
            0.025052500000000144,
            0.02502749999999998,
            0.025016499999999997,
            0.024997499999999784,
            0.024977000000000027,
        ]

        sodiumcitratepercentage = [
            0.0,
            13.333333333333336,
            18.333333333333336,
            21.666666666666668,
            30.000000000000004,
            11.666666666666666,
            33.333333333333336,
            48.33333333333334,
            46.666666666666664,
            53.33333333333334,
            55.00000000000001,
            56.66666666666668,
            63.33333333333334,
            68.33333333333333,
            71.66666666666667,
            75.0,
            76.66666666666667,
            83.33333333333334,
            85.00000000000001,
            86.66666666666667,
            93.33333333333333,
            93.33333333333333,
            96.66666666666669,
            100.0,
        ]

        fig1, axes1, dots1, _, xlabel1, ylabel1 = create_graph(
            phs, citrate_couplings, 'pH', 'ppm'
        )
        fig2, axes2, dots2, _, xlabel2, ylabel2 = create_graph(
            sodiumcitratepercentage,
            citrate_couplings,
            'Trisodium Citrate Ratio (%)',
            'ppm',
        )

        self.play(Create(axes1))
        self.add(xlabel1, ylabel1)
        self.play(Create(fig1), Create(dots1))
        self.wait(2)

        self.play(Transform(axes1, axes2))
        self.play(
            Transform(fig1, fig2),
            Transform(dots1, dots2),
            Transform(xlabel1, xlabel2),
        )
        self.wait(2)
