# `ipython-math-syntax`

Write textbook mathematical notation—implicit multiplication (`2x`), superscripts (`x²`), roots (`√x`, `∛x`), integrals (`∫₀¹ 3x² dx`), derivatives (`d/dx`, `∂²/∂x∂y`, `∂ₓ`, `f′(x)`, `ẋ`), vector calculus (`∇`, `∇⋅`, `∇×`, `∇²`), Einstein summation (`ein(Aᵢⱼ Bⱼₖ)`), Kronecker & Levi-Civita tensors (`δᵢⱼ`, `εᵢⱼₖ`), Dirac bra-kets (`|0⟩`, `⟨+|σ_x|+⟩`), and more—directly inside **Jupyter Notebooks**, **JupyterLab**, and **IPython**.
<img width="1280" height="720" alt="image" src="https://github.com/user-attachments/assets/69a6e011-dcbe-4194-8643-61cfb4e94ee0" />

---

## Installation

```bash
pip install .
```

Or install directly from GitHub:

```bash
pip install git+https://github.com/YOUR_USERNAME/ipython-math-syntax.git
```

## Quick Start

In any Jupyter Notebook or IPython session, load the extension in one line:

```python
%load_ext math_syntax
```

To unload at any time:

```python
%unload_ext math_syntax
```

---

## Supported Notation Reference

| Category | Math Syntax | What It Computes |
| :--- | :--- | :--- |
| **Implicit Multiplication** | `2x`, `3π`, `2(x+1)`, `(a+b)(c+d)`, `x²y` | `2*x`, `3*π`, `2*(x+1)`, `(a+b)*(c+d)`, `x**(2)*y` (preserves `1e-5`, `0xFF`) |
| **Superscripts & Subscripts** | `x²`, `ℯ⁻ˣ²`, `2ⁿ⁺¹`, `x₁`, `aᵢ` | `x**(2)`, `exp(-x**2)`, `2**(n+1)`, `x_1`, `a_i` |
| **Trig Powers & Inverses** | `sin²(θ) + cos²(θ)`, `sin⁻¹(1/2)`, `h⁻¹(y)` | `(sin(θ))**2 + (cos(θ))**2`, `asin(1/2)`, symbolic inverse $h^{-1}(y)$ |
| **Roots & Postfix Ops** | `√x`, `∛x`, `∜x`, `n!`, `n!!`, `30°` | `sqrt(x)`, `cbrt(x)`, `fourth_root(x)`, `factorial(n)`, `factorial2(n)`, `radians(30)` |
| **Indefinite & Definite Integrals** | `∫ x² dx`, `∫₀¹ 3x² dx`, `∫₋∞^∞ ℯ⁻ˣ² dx`, `∬ (6x⋅y²) dx dy` | Exact symbolic integrals via SymPy (`sp.integrate`) |
| **Evaluation Bar** | `(x³ / 3)\|₀³`, `(-cos(x))\|₀^π` | Fundamental theorem evaluation $F(b) - F(a)$ |
| **Derivatives** | `d/dx(sin(x²))`, `∂²/∂x∂y(x³y²)`, `∂ₓᵧ(x²y³)`, `g′(x)`, `ẋ`, `ẍ` | Leibniz, subscript PDE, Lagrange prime, and Newton time-dot derivatives |
| **Vector Calculus** | `∇f`, `∇⋅F`, `∇×F`, `∇²f` (or `∆f`), `jacobian`, `hessian` | Gradient, divergence, 3D curl, Laplacian, Jacobian, Hessian |
| **Prefix Sums & Products** | `∑ᵢ₌₁ⁿ i²`, `∑ₙ₌₁^∞ (1/n²)`, `∏ₖ₌₁⁵ k`, `∑_{n=1}^{∞} (1/2ⁿ)` | Symbolic/numerical $\sum$, $\prod$, $\bigcup$, $\bigcap$ |
| **Limits & Taylor Series** | `lim((1 + 1/n)ⁿ, n → ∞)`, `lim(1/x, x → 0⁺)`, `taylor(sin(x))` | Symbolic one/two-sided limits and Taylor expansions |
| **Integral Transforms & ODEs** | `ℒ(sin(2t))`, `ℒ⁻¹(1/(s²+1))`, `ℱ(ℯ^(-x²))`, `solve(y″ + y ≟ 0)` | Laplace, Inverse Laplace, Fourier transforms, and ODE/algebraic solver |
| **Tensors & Einstein Sum** | `δᵢⱼ`, `εᵢⱼₖ`, `ein(Aᵢᵢ)`, `ein(Aᵢⱼ Bⱼₖ)`, `ein(εᵢⱼₖ uⱼ vₖ)` | Kronecker delta, Levi-Civita, and automatic index contraction |
| **Products & Algebra** | `u × v`, `A ⊗ B`, `u ∧ v`, `A ⊕ B`, `u ⊛ v`, `A ⊡ B` | 3D cross / Cartesian product, Kronecker product, wedge product, direct sum, convolution, double contraction |
| **Quantum Bra-Kets** | `\|0⟩`, `\|+⟩`, `⟨0\|1⟩`, `\|0⟩⟨0\|`, `⟨+\|σ_x\|+⟩`, `⟦σ_z, σ_x⟧` | Qubit kets/bras, projectors, expectation values, commutators |
| **Overloaded Bars & Norms** | `\|-7\|`, `\|3 + 4𝕚\|`, `\|A\|`, `\|{1,2,3}\|`, `‖v‖`, `‖v‖₁`, `‖v‖∞` | Abs, complex modulus, matrix determinant, set cardinality, and $L_p$ norms |
| **Combinatorics & Number Theory**| `⁵C₂`, `⁵P₂`, `17 ≡ 2 (mod 5)`, `7 ∣ 42`, `8 ⊥ 15` | Binomial, k-permutation, modular congruence, divisibility, coprimality / orthogonality |
| **Sets, Domains & Quantifiers**| `√2 ∈ ℝ ∧ √2 ∉ ℚ`, `{x² ∣ x ∈ A : x > 2}`, `∀x ∈ A: P(x)`, `∃`, `∄` | Number domains ($\mathbb{N}, \mathbb{Z}, \mathbb{Q}, \mathbb{R}, \mathbb{C}$), set-builder comprehension, quantifiers |

---

## Example Showcase

```python
%load_ext math_syntax

# 1. Calculus & Integrals
print(∫₀¹ 3x² dx)                  # 1
print(∫₋∞^∞ ℯ⁻ˣ² dx)               # sqrt(pi)
print(∂²/∂x∂y(x³y²))               # 6*x**2*y

# 2. Summations & Einstein Contraction
print(sp.factor(∑ᵢ₌₁ⁿ i²))         # n*(n + 1)*(2*n + 1)/6
print(∑ₙ₌₁^∞ (1/n²))               # pi**2/6
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])
print(ein(Aᵢⱼ Bⱼₖ))                # Matrix([[19, 22], [43, 50]])

# 3. Differential Equations & Transforms
print(solve(y″ + y ≟ 0))           # Eq(y(x), C1*sin(x) + C2*cos(x))
print(ℒ(sin(2t)))                  # 2/(s**2 + 4)

# 4. Quantum Mechanics
σ_x = Matrix([[0, 1], [1, 0]])
print(⟨+|σ_x|+⟩)                   # 1
```
