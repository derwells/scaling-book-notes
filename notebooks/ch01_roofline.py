import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    return np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Roofline exercises

    Problems from [Part 1](https://jax-ml.github.io/scaling-book/roofline).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Q1 — int8 matmul

    Say we want to do the matmul $X[B, D] \cdot_D Y[D, F] \to Z[B, F]$ in int8 precision
    (1 byte per parameter) instead of bfloat16 (2 bytes per parameter) since TPUs/GPUs can
    do matmuls faster in lower precision.

    1. How many bytes need to be loaded from memory? How many need to be written back to memory?
    2. How many total OPs are performed?
    3. What is the arithmetic intensity?
    4. What is a roofline estimate for $T_\text{math}$ and $T_\text{comms}$? What are reasonable upper and lower bounds for the runtime of the whole operation?

    Assume our HBM bandwidth is 8.2e11 bytes/s and our int8 peak OPs/s is 3.94e14 (about 2x bfloat16).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **1.1** $\text{bytes loaded} = BD + DF$, $\text{bytes written} = BF$

    **1.2** $BF$ elements, each a dot product of length $D$: $D$ muls + $(D-1)$ adds

    $$\text{ops} = BF(2D - 1)$$

    **1.3**

    $$I = \frac{BF(2D - 1)}{BD + DF + BF}$$

    for $B \ll D, F$:

    $$I \approx 2B$$

    **1.4**

    $$T_\text{math} = \frac{BF(2D-1)}{3.94 \times 10^{14}}, \quad T_\text{comms} = \frac{BD + DF + BF}{8.2 \times 10^{11}}$$

    $$T_\text{lower} = \max(T_\text{math}, T_\text{comms}), \quad T_\text{upper} = T_\text{math} + T_\text{comms}$$
    """)
    return


@app.cell
def _():
    int8_ops = 3.94e14
    hbm_bw = 8.2e11
    # same B > 240 as bf16 — doubled ops/s cancels halved bytes
    print(f"critical intensity = {int8_ops / hbm_bw:.0f}")
    print(f"B > {int8_ops / hbm_bw / 2:.0f}")
    return hbm_bw, int8_ops


@app.cell
def _(hbm_bw, int8_ops):
    B, D, F = 128, 8192, 8192
    t_math = B * F * (2 * D - 1) / int8_ops
    t_comms = (B * D + D * F + B * F) / hbm_bw
    print(f"B={B}, D={D}, F={F}")
    print(f"T_math  = {t_math*1e3:.3f} ms")
    print(f"T_comms = {t_comms*1e3:.3f} ms")
    print(f"lower = {max(t_math, t_comms)*1e3:.3f} ms")
    print(f"upper = {(t_math + t_comms)*1e3:.3f} ms")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Q2 — int8 weights + bf16 activations

    In practice we often do different weight vs. activation quantization, so we might store
    our weights in very low precision but keep activations (and compute) in a higher precision.
    Say we want to quantize our weights in int8 but keep activations (and compute) in bfloat16.
    At what batch size do we become compute bound? Assume 1.97e14 bfloat16 FLOPs/s.

    Specifically: `bf16[B, D] * int8[D, F] -> bf16[B, F]` where $B$ is the batch size.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    bytes: $2BD + DF + 2BF$, FLOPs: $2BDF$

    $$I = \frac{2BDF}{2BD + DF + 2BF} \approx 2B$$
    """)
    return


@app.cell
def _():
    bf16_flops = 1.97e14
    bw = 8.2e11
    # B > hw_intensity / 2
    print(f"hw intensity = {bf16_flops / bw:.0f}")
    print(f"B > {bf16_flops / bw / 2:.0f}")
    return bf16_flops, bw


@app.cell
def _(mo):
    mo.md(r"""
    ## Q3 — Roofline plot

    Taking the setup from Question 2, make a roofline plot of peak FLOPs/s vs. $B$
    for $F = D = 4096$ and $F = D = 1024$. Use the exact number of bytes loaded,
    not an approximation.
    """)
    return


@app.cell
def _():
    # your work here
    ...
    return


@app.cell
def _(bf16_flops, bw, np, plt):
    def _roofline(B: np.ndarray, D, F) -> np.ndarray:
        """Returns FLOPs/s"""
        flops = 2*B*D*F
        bytes_sent = 2*B*D + D*F + 2*B*F
        t_math = flops / bf16_flops
        t_comms = bytes_sent / bw
        return flops / np.maximum(t_math, t_comms)

    fig, ax = plt.subplots()
    _Bs = np.arange(1, 1024)
    ax.plot(_Bs, _roofline(_Bs, 4096, 4096), label="D=F=4096")
    ax.plot(_Bs, _roofline(_Bs, 1024, 1024), label="D=F=1024")
    ax.set_xlabel("B")
    ax.set_ylabel("FLOPs/s")
    ax.legend()
    fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Q4 — per-batch weight matrices

    What if we wanted to perform $\text{int8}[B, D] \cdot_D \text{int8}[B, D, F] \to \text{int8}[B, F]$
    where we imagine having a different matrix for each batch element. What is the arithmetic
    intensity of this operation?
    """)
    return


@app.cell
def _():
    # your work here
    ...
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Q5 — Memory rooflines for GPUs

    Using the [spec sheet provided by NVIDIA for the H100 SXM](https://www.nvidia.com/en-us/data-center/h100/),
    calculate the batch size at which a bfloat16 matrix multiplication will become compute-bound.
    Note that the Tensor Core FLOPs numbers are twice the true value since they're only
    achievable with structured sparsity.
    """)
    return


@app.cell
def _():
    # your work here
    ...
    return


if __name__ == "__main__":
    app.run()
