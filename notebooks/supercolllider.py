import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import threading

    # FIXED: Supriya uses `Thread` which breaks marimo
    import marimo as mo
    threading.Thread = mo.Thread

    import supriya as sc
    import supriya.ugens as scu
    return (sc,)


@app.cell
def _(sc):
    server = sc.Server().quit()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
