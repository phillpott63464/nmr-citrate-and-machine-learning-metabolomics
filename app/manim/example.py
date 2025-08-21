from manim import *


class PlotPoints(Scene):
    def construct(self):
        phs = [2.65, 3.04, 2.99, 3.31, 3.33, 2.96, 3.83, 3.65, 4.36, 4.61, 4.7, 4.9, 5.13, 5.32, 5.48, 5.59, 5.79, 6.0, 6.8, 6.3, 6.51, 6.55, 7.09, 7.53]
        # pts = [[1, 2], [3, 4]]  # your array of [x,y]
        citrate_couplings = [0.02620699999999987, 0.026074500000000222, 0.02601449999999983, 0.02594950000000007, 0.025904499999999997, 0.0261015, 0.02575700000000003, 0.025643000000000082, 0.025580500000000228, 0.0254915, 0.02543150000000005, 0.02535200000000004, 0.025285500000000072, 0.025217000000000045, 0.0251595, 0.025150999999999923, 0.025144500000000125, 0.02511449999999993, 0.025070499999999774, 0.025052500000000144, 0.02502749999999998, 0.025016499999999997, 0.024997499999999784, 0.024977000000000027]
        pts = zip(phs, citrate_couplings)

        # If you want Axes to place coordinates nicely:
        axes = Axes(
            x_range=[min(phs), max(phs), (max(phs)-min(phs)) / 10],
            y_range=[min(citrate_couplings), max(citrate_couplings), (max(citrate_couplings) - min(citrate_couplings)) / 10],
            # x_length=6,
            # y_length=4,
            tips=False,
            axis_config={"include_numbers": True},
        ).to_edge(DOWN)
        self.add(axes)

        # Convert data points to coordinates in the axes' space
        coord_points = [axes.coords_to_point(x, y) for x, y in pts]

        # Create polyline connecting points
        poly = VMobject()
        poly.set_points_as_corners(coord_points)
        poly.set_stroke(BLUE, 3)

        # Optional: show small dots at each data point
        # dots = VGroup(*[Dot(p, radius=0.06, color=YELLOW) for p in coord_points])

        # Optional labels for each point
        labels = VGroup(*[
            MathTex(f"({x},{y})").scale(0.5).next_to(pt, UR, buff=0.06)
            for (x, y), pt in zip(pts, coord_points)
        ])

        # self.play(Create(poly), FadeIn(dots), Write(labels))
        self.play(Create(poly), Write(labels))
        self.wait(2)

        nacitrate = [0.0, 13.333333333333336, 18.333333333333336, 21.666666666666668, 30.000000000000004, 11.666666666666666, 33.333333333333336, 48.33333333333334, 46.666666666666664, 53.33333333333334, 55.00000000000001, 56.66666666666668, 63.33333333333334, 68.33333333333333, 71.66666666666667, 75.0, 76.66666666666667, 83.33333333333334, 85.00000000000001, 86.66666666666667, 93.33333333333333, 93.33333333333333, 96.66666666666669, 100.0]

        pts = zip(nacitrate, citrate_couplings)

        nacitrate_citratecouplings_axes = Axes(
            x_range=[min(nacitrate), max(nacitrate), (max(nacitrate)-min(nacitrate)) / 10],
            y_range=[min(citrate_couplings), max(citrate_couplings), (max(citrate_couplings) - min(citrate_couplings)) / 10],
            # x_length=6,
            # y_length=4,
            tips=False,
            axis_config={"include_numbers": True},
        ).to_edge(DOWN)
        # self.add(axes)

        

        coord_points_citrate = [nacitrate_citratecouplings_axes.coords_to_point(x, y) for x, y in pts]
        nacitrate_citratecouplings_fig = VMobject()
        nacitrate_citratecouplings_fig.set_points_as_corners(coord_points_citrate)
        nacitrate_citratecouplings_fig.set_stroke(BLUE, 3)

        self.play(Transform(poly, nacitrate_citratecouplings_fig), Transform(axes, nacitrate_citratecouplings_axes))
        self.wait(2)
