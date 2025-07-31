import marimo

__generated_with = '0.14.13'
app = marimo.App(width='medium')


@app.cell
def _():
    import marimo as mo
    import torch

    print(torch.version.hip)  # Should show HIP version if ROCm build
    print(torch.backends.cuda.is_built())  # Should be False for ROCm
    print(torch.cuda.device_count())  # Should show GPU count if detected
    return


if __name__ == '__main__':
    app.run()
