# pyright: basic

from matplotlib.figure import Figure


def simple_plot():
    fig = Figure(figsize=(6, 4), dpi=100)
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    ax.plot([0, 1, 2], [2, 3, 1])
    ax.set_title("Pure‐OO in Jupyter")

    return fig


def dual_xaxis_plot():
    fig = Figure(figsize=(6, 4), dpi=100)
    ax1 = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    ax2 = ax1.twinx()

    # variables to plot
    x = [0, 1, 2, 3, 4]
    y1 = [10, 30, 20, 40, 30]
    y2 = [100, 80, 60, 40, 20]

    # plot on the left y-axis
    ax1.plot(x, y1, "g-", label="y1 data")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y1", color="g")
    ax1.tick_params(axis="y", colors="g")

    # plot on the left y-axis
    ax2.plot(x, y2, "b--", label="y2 data")
    ax2.set_ylabel("y2", color="b")
    ax2.tick_params(axis="y", colors="b")

    return fig


def grid_plot():
    # create a Figure with constrained_layout turned on
    fig = Figure(figsize=(8, 6), constrained_layout=True)

    # make a 2×2 grid, sharing both x- and y-axes
    axes = fig.subplots(2, 2, sharex=True, sharey=True)

    # now plot into each Axes
    x = [0, 1, 2, 3, 4]
    for i, ax in enumerate(axes.flat, start=1):
        y = [v * i for v in x]
        ax.plot(x, y)
        ax.set_title(f"plot {i}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    return fig
