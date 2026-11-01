"""
ipython-math-syntax: Write textbook mathematical notation directly in Jupyter / IPython.
"""

import re
import math
import cmath
import operator
import itertools
import numpy as np
import sympy as sp

__version__ = "0.1.0"
_MATH_SYNTAX_SETUP_GUARD_ = True

# --- 1. Symbolic Variables, Constants & Number Domains ---
x, y, z, t = sp.symbols('x y z t', real=True)
k, m, n, i, j = sp.symbols('k m n i j', integer=True)
s = sp.Symbol('s')
θ, ω = sp.symbols('θ ω', real=True)

π = sp.pi
τ = 2 * sp.pi
φ = ϕ = sp.GoldenRatio
ℏ = sp.Symbol('ℏ', positive=True)

_E_ = sp.E
_I_ = sp.I
_OO_ = sp.oo
_EMPTYSET_ = frozenset()
_NATURALS_  = sp.S.Naturals
_INTEGERS_  = sp.S.Integers
_RATIONALS_ = sp.S.Rationals
_REALS_     = sp.S.Reals
_COMPLEXES_ = sp.S.Complexes
_RE_, _IM_  = sp.re, sp.im
_PARTIAL_   = sp.diff
_DIR_POS_, _DIR_NEG_ = '+', '-'
_KET_PLUS_, _KET_MINUS_ = '+', '-'

# --- 2. Smart Elementary & Combinatorial Functions ---
class _SmartPowFn:
    def __init__(self, fn, power):
        self.fn = fn
        self.power = power
    def __call__(self, *args):
        return (self.fn(*args)) ** self.power
    def __mul__(self, other):
        return (self.fn(other)) ** self.power

class _SmartFn:
    def __init__(self, sp_fn, np_fn=None, inv_fn=None):
        self.sp_fn = sp_fn
        self.np_fn = np_fn
        self.inv_fn = inv_fn
    def __call__(self, *args):
        if len(args) == 1 and isinstance(args[0], np.ndarray) and self.np_fn is not None:
            return self.np_fn(args[0])
        return self.sp_fn(*args)
    def __pow__(self, power):
        if power == -1 and self.inv_fn is not None:
            return self.inv_fn
        return _SmartPowFn(self, power)

asin  = _SmartFn(sp.asin, np.arcsin)
acos  = _SmartFn(sp.acos, np.arccos)
atan  = _SmartFn(sp.atan, np.arctan)
asinh = _SmartFn(sp.asinh, np.arcsinh)
acosh = _SmartFn(sp.acosh, np.arccosh)
atanh = _SmartFn(sp.atanh, np.arctanh)

sqrt  = _SmartFn(sp.sqrt, np.sqrt)
cbrt  = _SmartFn(sp.cbrt, np.cbrt)
fourth_root = lambda a: sp.root(a, 4) if isinstance(a, (int, sp.Basic)) else a ** 0.25
sin   = _SmartFn(sp.sin, np.sin, asin)
cos   = _SmartFn(sp.cos, np.cos, acos)
tan   = _SmartFn(sp.tan, np.tan, atan)
sec   = _SmartFn(sp.sec)
csc   = _SmartFn(sp.csc)
cot   = _SmartFn(sp.cot)
atan2 = sp.atan2
sinh  = _SmartFn(sp.sinh, np.sinh, asinh)
cosh  = _SmartFn(sp.cosh, np.cosh, acosh)
tanh  = _SmartFn(sp.tanh, np.tanh, atanh)
exp   = _SmartFn(sp.exp, np.exp)
ln    = _SmartFn(sp.log, np.log, exp)
log   = _SmartFn(sp.log, np.log, exp)
log10 = lambda a: sp.log(a, 10)
log2  = lambda a: sp.log(a, 2)
floor = sp.floor
ceil  = sp.ceiling
factorial   = sp.factorial
factorial2  = sp.factorial2
binomial    = sp.binomial
permutation = lambda n_, k_: sp.factorial(n_) / sp.factorial(n_ - k_)
gcd, lcm = sp.gcd, sp.lcm
conjugate = sp.conjugate
radians = lambda d: d * sp.pi / 180 if isinstance(d, (int, sp.Basic)) else math.radians(d)
degrees = lambda r: r * 180 / sp.pi if isinstance(r, sp.Basic) else math.degrees(r)

# --- 3. Calculus, Evaluation Bar, Transforms, Taylor, ODEs & Series ---
def _integrate(expr, *bounds):
    return sp.integrate(expr, *bounds)

def _eval_bar(expr, a, b, var=None):
    if not isinstance(expr, sp.Basic):
        return expr
    if var is None:
        syms = [v for v in (x, t, y, z, θ) if v in expr.free_symbols]
        var = syms[0] if syms else (sorted(expr.free_symbols, key=lambda s_: s_.name)[0] if expr.free_symbols else x)
    return expr.subs(var, b) - expr.subs(var, a)

def taylor(expr, var=x, at=0, order=5):
    return sp.series(expr, var, at, order).removeO()

class _DiffOp:
    _op_priority = 1000.0
    __array_ufunc__ = None
    def __init__(self, *vars_and_orders):
        self.args = vars_and_orders
    def __call__(self, expr):
        return sp.diff(expr, *self.args)
    def __mul__(self, expr):
        return sp.diff(expr, *self.args)

def _prime_diff(f, order=1):
    if isinstance(f, sp.Symbol) and f != x:
        fn_cls = sp.Function(f.name)
        return sp.diff(fn_cls(x), x, order)
    if isinstance(f, sp.Basic):
        syms = sorted(f.free_symbols, key=lambda s_: s_.name)
        var = syms[0] if syms else x
        return sp.diff(f, var, order)
    if callable(f):
        return sp.diff(f(x), x, order)
    return sp.diff(f, x, order)

def _prime_eval(f, order, val):
    if callable(f):
        _s = sp.Symbol('_s', real=True)
        d_expr = sp.diff(f(_s), _s, order)
        if isinstance(val, sp.Basic):
            return d_expr.subs(_s, val)
        res = d_expr.subs(_s, val).evalf()
        return float(res) if res.is_real else complex(res)
    if isinstance(f, sp.Basic):
        syms = sorted(f.free_symbols, key=lambda s_: s_.name)
        var = syms[0] if syms else (val if isinstance(val, sp.Symbol) else x)
        d_expr = sp.diff(f, var, order)
        return d_expr.subs(var, val)
    return sp.diff(f, x, order)

def _dot_diff(f, order=1):
    if isinstance(f, sp.Symbol) and f != t:
        fn_cls = sp.Function(f.name)
        return sp.diff(fn_cls(t), t, order)
    return sp.diff(f, t, order)

class _LaplaceOp:
    def __call__(self, expr, t_var=t, s_var=s):
        return sp.laplace_transform(expr, t_var, s_var, noconds=True)
    def inv(self, expr, s_var=s, t_var=t):
        return sp.inverse_laplace_transform(expr, s_var, t_var)

class _FourierOp:
    def __call__(self, expr, x_var=x, k_var=ω):
        return sp.fourier_transform(expr, x_var, k_var)
    def inv(self, expr, k_var=ω, x_var=x):
        return sp.inverse_fourier_transform(expr, k_var, x_var)

_LAPLACE_ = _LaplaceOp()
_FOURIER_ = _FourierOp()

def _func_inv(f):
    if hasattr(f, 'inv') and callable(f.inv):
        return f.inv
    if isinstance(f, _SmartFn) and f.inv_fn is not None:
        return f.inv_fn
    if isinstance(f, sp.MatrixBase):
        return lambda v: f.inv() * v
    if callable(f):
        def _inv_eval(val=x):
            _s = sp.Symbol('_s', real=True)
            sols = sp.solve(sp.Eq(f(_s), val), _s)
            return sols[0] if len(sols) == 1 else sols
        return _inv_eval
    raise TypeError(f"Object {f} is not invertible as a function.")

def solve(eq, *args, **kwargs):
    if isinstance(eq, sp.Basic) and eq.has(sp.Derivative):
        derivs = list(eq.atoms(sp.Derivative))
        for d in derivs:
            fn = d.expr
            if hasattr(fn, 'func'):
                fname = getattr(fn.func, '__name__', str(fn.func))
                eq = eq.subs(sp.Symbol(fname, real=True), fn)
                eq = eq.subs(sp.Symbol(fname), fn)
        return sp.dsolve(eq, *args, **kwargs)
    return sp.solve(eq, *args, **kwargs)

def lim(expr, var, to, direction=None):
    return sp.limit(expr, var, to) if direction is None else sp.limit(expr, var, to, direction)

def Σ(expr_or_iter, *bounds):
    return sp.summation(expr_or_iter, *bounds) if bounds else sum(expr_or_iter)

def Π(expr_or_iter, *bounds):
    return sp.product(expr_or_iter, *bounds) if bounds else math.prod(expr_or_iter)

def _big_union(expr_or_iter, *bounds):
    if bounds:
        var, a, b = bounds[0]
        return set().union(*(expr_or_iter.subs(var, v) if isinstance(expr_or_iter, sp.Basic) else expr_or_iter(v) for v in range(int(a), int(b) + 1)))
    return set().union(*expr_or_iter)

def _big_intersect(expr_or_iter, *bounds):
    if bounds:
        var, a, b = bounds[0]
        sets = [expr_or_iter.subs(var, v) if isinstance(expr_or_iter, sp.Basic) else expr_or_iter(v) for v in range(int(a), int(b) + 1)]
    else:
        sets = list(expr_or_iter)
    return set.intersection(*(set(s_) for s_ in sets)) if sets else set()

_BIGUNION_ = _big_union
_BIGINTER_ = _big_intersect
_FORALL_   = all
_EXISTS_   = any
_NEXISTS_  = lambda it: not any(it)

# --- 4. Vector Calculus (∇, ∇⋅, ∇×, ∇² / ∆, Jacobian, Hessian) ---
def _default_coords(expr):
    if isinstance(expr, sp.Basic):
        c = [v for v in (x, y, z) if v in expr.free_symbols]
        return c if c else sorted(expr.free_symbols, key=lambda s_: s_.name) or [x]
    return [x, y, z]

def _grad(f, coords=None):
    coords = coords or _default_coords(f)
    return sp.Matrix([sp.diff(f, v) for v in coords])

def _div(F, coords=(x, y, z)):
    return sum(sp.diff(comp, var) for comp, var in zip(F, coords))

def _curl(F, coords=(x, y, z)):
    if len(F) != 3:
        raise ValueError("3D Curl requires a 3-component vector.")
    X, Y, Z = coords[:3]
    return sp.Matrix([
        sp.diff(F[2], Y) - sp.diff(F[1], Z),
        sp.diff(F[0], Z) - sp.diff(F[2], X),
        sp.diff(F[1], X) - sp.diff(F[0], Y)
    ])

def _laplacian(f, coords=None):
    coords = coords or _default_coords(f)
    return sum(sp.diff(f, v, 2) for v in coords)

def jacobian(F, coords=None):
    M = sp.Matrix(F)
    if coords is None:
        free = set().union(*(el.free_symbols for el in M if isinstance(el, sp.Basic)))
        coords = [v for v in (x, y, z, t) if v in free] or sorted(free, key=lambda s_: s_.name)
    return M.jacobian(coords)

def hessian(f, coords=None):
    coords = coords or _default_coords(f)
    return sp.hessian(f, coords)

# --- 5. Tensors, Einstein Summation, Quantum Bra-Kets & Linear Algebra ---
δ = sp.KroneckerDelta
ε = ϵ = sp.LeviCivita
Array = sp.Array
Matrix = sp.Matrix

def contract(tensor, *axes):
    return sp.tensorcontraction(sp.Array(tensor), *axes)

def _einsum(subscripts, *operands):
    if all(isinstance(op, np.ndarray) for op in operands):
        return np.einsum(subscripts, *operands)
    in_subs = subscripts.split(',')
    dim = 3
    for op in operands:
        if isinstance(op, (list, tuple, sp.MatrixBase, sp.NDimArray, np.ndarray)):
            dim = op.shape[0] if hasattr(op, 'shape') else len(op)
            break
    mats = []
    for op, s_ in zip(operands, in_subs):
        if op is ε:
            arr = sp.Array(
                [sp.LeviCivita(*(idx + 1 for idx in indices)) for indices in itertools.product(range(dim), repeat=len(s_))],
                (dim,) * len(s_)
            )
            mats.append(arr)
        elif op is δ:
            mats.append(sp.Array(sp.eye(dim)))
        else:
            mats.append(sp.Array(op))
    counts = {}
    for s_ in in_subs:
        for ch in s_:
            counts[ch] = counts.get(ch, 0) + 1
    free_idx = [ch for ch in dict.fromkeys("".join(in_subs)) if counts[ch] == 1]
    sum_idx  = [ch for ch in dict.fromkeys("".join(in_subs)) if counts[ch] > 1]

    def _eval_at(free_vals):
        idx_map = dict(zip(free_idx, free_vals))
        total = 0
        for sum_vals in itertools.product(range(dim), repeat=len(sum_idx)):
            idx_map.update(zip(sum_idx, sum_vals))
            term = 1
            for arr, s_ in zip(mats, in_subs):
                coord = tuple(idx_map[c] for c in s_)
                term = term * (arr[coord] if len(coord) > 1 else arr[coord[0]])
            total = total + term
        return sp.simplify(total) if isinstance(total, sp.Basic) else total

    if not free_idx:
        return _eval_at(())
    if len(free_idx) == 1:
        return [_eval_at((i_,)) for i_ in range(dim)]
    if len(free_idx) == 2:
        return sp.Matrix([[_eval_at((i_, j_)) for j_ in range(dim)] for i_ in range(dim)])
    return sp.Array([_eval_at(vals) for vals in itertools.product(range(dim), repeat=len(free_idx))], (dim,) * len(free_idx))

def _ket(v):
    if isinstance(v, int) and v in (0, 1):
        return sp.Matrix([[1], [0]]) if v == 0 else sp.Matrix([[0], [1]])
    if isinstance(v, str) and v in ('+', '-'):
        return sp.Matrix([[1], [1]]) / sp.sqrt(2) if v == '+' else sp.Matrix([[1], [-1]]) / sp.sqrt(2)
    if isinstance(v, sp.MatrixBase):
        return v if v.cols == 1 else v.T
    return sp.Matrix(list(v))

def _bra(v):
    return _ket(v).H

def inner(u, v=None):
    if v is None and isinstance(u, tuple) and len(u) == 2:
        u, v = u
    if isinstance(u, (int, str)) or isinstance(v, (int, str)):
        return (_bra(u) * _ket(v))[0, 0]
    return sum(sp.conjugate(a) * b for a, b in zip(u, v))

def _expectation(u, A, v):
    return (_bra(u) * sp.Matrix(A) * _ket(v))[0, 0]

def _abs_det_card(obj):
    if isinstance(obj, sp.MatrixBase):
        return obj.det()
    if isinstance(obj, (set, frozenset)):
        return len(obj)
    if isinstance(obj, sp.Set):
        return len(obj) if isinstance(obj, sp.FiniteSet) else obj.measure
    if isinstance(obj, np.ndarray) and obj.ndim == 2:
        return np.linalg.det(obj)
    return sp.Abs(obj)

def _commutator(A, B):
    if isinstance(A, (list, tuple)): A = sp.Matrix(A)
    if isinstance(B, (list, tuple)): B = sp.Matrix(B)
    return A * B - B * A

def _anticommutator(A, B):
    if isinstance(A, (list, tuple)): A = sp.Matrix(A)
    if isinstance(B, (list, tuple)): B = sp.Matrix(B)
    return A * B + B * A

def _double_contract(A, B):
    ta, tb = sp.Array(A), sp.Array(B)
    return sum(a * b for a, b in zip(ta, tb))

def _cross_or_mul(a, b):
    if isinstance(a, (set, frozenset)) and isinstance(b, (set, frozenset)):
        return {(x_, y_) for x_ in a for y_ in b}
    if isinstance(a, sp.Set) and isinstance(b, sp.Set):
        return sp.ProductSet(a, b)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == 3 and len(b) == 3:
        return [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]
        ]
    return a * b

def _tensor_prod(a, b):
    if isinstance(a, sp.MatrixBase) and isinstance(b, sp.MatrixBase):
        return sp.kronecker_product(a, b)
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return np.kron(a, b)
    if isinstance(a, (list, tuple, sp.NDimArray)) or isinstance(b, (list, tuple, sp.NDimArray)):
        return sp.tensorproduct(sp.Array(a), sp.Array(b))
    return sp.tensorproduct(a, b)

def _wedge_or_and(a, b):
    if isinstance(a, (list, tuple, sp.MatrixBase, sp.NDimArray, np.ndarray)) and \
       isinstance(b, (list, tuple, sp.MatrixBase, sp.NDimArray, np.ndarray)):
        ta, tb = sp.Array(a), sp.Array(b)
        return sp.tensorproduct(ta, tb) - sp.tensorproduct(tb, ta)
    if isinstance(a, sp.Basic) or isinstance(b, sp.Basic):
        return sp.And(a, b)
    return bool(a and b)

def _direct_sum_or_xor(a, b):
    if isinstance(a, sp.MatrixBase) and isinstance(b, sp.MatrixBase):
        return sp.diag(a, b)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return sp.diag(sp.Matrix(a), sp.Matrix(b))
    return operator.xor(a, b)

def _transpose(A):
    if hasattr(A, 'T'):
        return A.T
    return sp.Matrix(A).T

def _dagger(A):
    if hasattr(A, 'H'):
        return A.H
    if isinstance(A, np.ndarray):
        return A.conj().T
    if isinstance(A, (list, tuple)):
        return sp.Matrix(A).H
    return sp.conjugate(A)

def _norm(v, p=2):
    if isinstance(v, sp.MatrixBase):
        return v.norm(p)
    if isinstance(v, (list, tuple, sp.NDimArray, np.ndarray)):
        if p == sp.oo or p == math.inf:
            return max(sp.Abs(el) for el in v)
        return sp.simplify(
            sum(sp.Abs(el) ** p for el in v) ** (sp.Rational(1, p) if isinstance(p, int) else 1 / p)
        )
    return sp.Abs(v)

det = lambda A: A.det() if isinstance(A, sp.MatrixBase) else sp.Matrix(A).det()
tr  = lambda A: A.trace() if isinstance(A, sp.MatrixBase) else sp.Matrix(A).trace()

# --- 6. Infix Operator Infrastructure (±, ∓, ×, ⊗, ⊕, ⋀, ⊛, ⊡, ≈, ≟, ∼, ∝, ⊥, ∥) ---
class _PMTuple(tuple):
    _op_priority = 1000.0
    __array_ufunc__ = None
    def __add__(self, o):      return _PMTuple(a + o for a in self)
    def __radd__(self, o):     return _PMTuple(o + a for a in self)
    def __sub__(self, o):      return _PMTuple(a - o for a in self)
    def __rsub__(self, o):     return _PMTuple(o - a for a in self)
    def __mul__(self, o):      return _PMTuple(a * o for a in self)
    def __rmul__(self, o):     return _PMTuple(o * a for a in self)
    def __truediv__(self, o):  return _PMTuple(a / o for a in self)
    def __rtruediv__(self, o): return _PMTuple(o / a for a in self)
    def __pow__(self, o):      return _PMTuple(a ** o for a in self)

class _InfixOp:
    _op_priority = 1000.0
    __array_ufunc__ = None
    def __init__(self, func): self.func = func
    def __rlshift__(self, left): return _InfixLeft(self.func, left)
    def __rshift__(self, right): return self.func(None, right)

class _InfixLeft:
    _op_priority = 1000.0
    __array_ufunc__ = None
    def __init__(self, func, left): self.func, self.left = func, left
    def __rshift__(self, right): return self.func(self.left, right)

class _MulInfixOp:
    _op_priority = 1000.0
    __array_ufunc__ = None
    def __init__(self, func): self.func = func
    def __rmatmul__(self, left): return _MulInfixLeft(self.func, left)

class _MulInfixLeft:
    _op_priority = 1000.0
    __array_ufunc__ = None
    def __init__(self, func, left): self.func, self.left = func, left
    def __matmul__(self, right): return self.func(self.left, right)

def _approx_eq(a, b):
    try:
        if isinstance(a, sp.Basic) or isinstance(b, sp.Basic):
            diff = sp.simplify(a - b)
            if diff == 0:
                return True
            return cmath.isclose(complex(sp.N(a)), complex(sp.N(b)), rel_tol=1e-5, abs_tol=1e-8)
        return cmath.isclose(complex(a), complex(b), rel_tol=1e-5, abs_tol=1e-8)
    except Exception:
        return bool(np.allclose(np.asarray(a, dtype=complex), np.asarray(b, dtype=complex)))

def _ortho_or_coprime(a, b):
    if isinstance(a, (int, sp.Integer)) and isinstance(b, (int, sp.Integer)):
        return sp.gcd(a, b) == 1
    return sp.simplify(inner(a, b)) == 0

def _asymptotic_eq(f, g):
    syms = sorted((sp.sympify(f).free_symbols | sp.sympify(g).free_symbols), key=lambda s_: s_.name)
    var = syms[0] if syms else x
    return sp.limit(f / g, var, sp.oo) == 1

def _proportional(f, g):
    ratio = sp.simplify(sp.sympify(f) / sp.sympify(g))
    return len(ratio.free_symbols & {x, y, z, t}) == 0

_CROSS     = _MulInfixOp(_cross_or_mul)
_TENSOR    = _MulInfixOp(_tensor_prod)
_OPLUS     = _MulInfixOp(_direct_sum_or_xor)
_WEDGE     = _MulInfixOp(_wedge_or_and)
_CONVOLVE  = _MulInfixOp(lambda u, v: np.convolve(u, v))
_DCONTRACT = _MulInfixOp(_double_contract)

_PM = _InfixOp(lambda a, b: _PMTuple((-b, b) if a is None else (a - b, a + b)))
_MP = _InfixOp(lambda a, b: _PMTuple((b, -b) if a is None else (a + b, a - b)))
_APPROX   = _InfixOp(_approx_eq)
_SYMEQ    = _InfixOp(lambda a, b: sp.Eq(a, b))
_ASYMP    = _InfixOp(_asymptotic_eq)
_PROPTO   = _InfixOp(_proportional)
_IMPLIES  = _InfixOp(lambda a, b: sp.Implies(a, b))
_IFF      = _InfixOp(lambda a, b: sp.Equivalent(a, b))
_DIVIDES  = _InfixOp(lambda a, b: b % a == 0)
_NDIVIDES = _InfixOp(lambda a, b: b % a != 0)
_ORTHO    = _InfixOp(_ortho_or_coprime)
_PARALLEL = _InfixOp(lambda u, v: sp.Matrix([list(u), list(v)]).rank() <= 1)

# --- 7. Pre-Parser Character Tables & Transformation Pipeline ---
_SUP_CHARS = "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁽⁾ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ"
_IDENT_SUB_CHARS = "₀₁₂₃₄₅₆₇₈₉ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓᵧᵦᵨ"
_EXPR_SUB_CHARS  = _IDENT_SUB_CHARS + "₊₋₍₎₌"

_IDENT_START = r'[a-zA-Z_πτφϕθδεϵσρλμνξζηγκχψωΓΔΘΛΞΠΣΥΦΨΩℒℱ]'
_IDENT_CONT  = r'[a-zA-Z0-9_πτφϕθδεϵσρλμνξζηγκχψωΓΔΘΛΞΠΣΥΦΨΩℒℱ]'
_IDENT       = rf'{_IDENT_START}{_IDENT_CONT}*'

_SUP_MAP = str.maketrans(
    _SUP_CHARS,
    "0123456789+-()abcdefghijklmnoprstuvwxyz"
)
_SUB_MAP = str.maketrans(
    _EXPR_SUB_CHARS,
    "0123456789aehijklmnoprstuvxybr+-()="
)
_TOKEN_SPLIT = re.compile(
    r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|#[^\n]*)'
)
_BAL_PAREN = r'\((?:[^()]|\((?:[^()]|\([^()]*\))*\))*\)'
_BAL_BRACK = r'\[(?:[^\[\]]|\[(?:[^\[\]]|\[[^\[\]]*\])*\])*\]'
_BAL_BRACE = r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}'
_BAL_EXPR_NO_COMMA_COLON = rf'(?:{_BAL_PAREN}|{_BAL_BRACK}|{_BAL_BRACE}|[^,:\)\]\}}\n;])+'
_BAL_EXPR_NO_COMMA       = rf'(?:{_BAL_PAREN}|{_BAL_BRACK}|{_BAL_BRACE}|[^,\)\]\}}\n;])+'

def _clean_bound(s_):
    return s_.translate(_SUB_MAP).translate(_SUP_MAP)

def _clean_qstate(s_):
    st = s_.strip()
    if st == '+': return '_KET_PLUS_'
    if st in ('-', '−'): return '_KET_MINUS_'
    return s_

def _translate_sup(s_):
    s_ = re.sub(r'([ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ])([⁰¹²³⁴⁵⁶⁷⁸⁹]+)', r'\1**\2', s_)
    return s_.translate(_SUP_MAP)

def _parse_ein(expr_str):
    terms = re.findall(rf'({_IDENT})([{_IDENT_SUB_CHARS}]+)', expr_str)
    if not terms:
        return expr_str
    subs = ",".join(idx.translate(_SUB_MAP) for _, idx in terms)
    ops  = ", ".join(name for name, _ in terms)
    return f"_einsum('{subs}', {ops})"

def _transform_code_line(code):
    # A0. Einstein Summation macro: ein(Aᵢⱼ Bⱼₖ) or ein(εᵢⱼₖ uⱼ vₖ)
    code = re.sub(r'\bein\(([^()]+)\)', lambda m: _parse_ein(m.group(1)), code)

    # A1. Prefix Textbook Summations, Products, Unions, Intersections:
    _BIG_OP_MAP = {'∑': 'Σ', '∏': 'Π', '⋃': '_BIGUNION_', '⋂': '_BIGINTER_'}
    code = re.sub(
        rf'([∑∏⋃⋂])_\{{({_IDENT})\s*=\s*([^}}]+)\}}\^(?:\{{([^}}]+)\}}|([a-zA-Z0-9_πτφϕθ∞+\-]+))\s*({_BAL_PAREN}|[^\s,)\]\n]+)',
        lambda m: f"{_BIG_OP_MAP[m.group(1)]}({m.group(6)}, ({m.group(2)}, {_clean_bound(m.group(3))}, {_clean_bound(m.group(4) or m.group(5))}))",
        code
    )
    code = re.sub(
        rf'([∑∏⋃⋂])\s*([ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓᵧ])\s*₌\s*([{_EXPR_SUB_CHARS}∞]+)(?:\^([a-zA-Z0-9_πτφϕθ∞+\-]+)|([{_SUP_CHARS}πτℯ∞]+))\s*({_BAL_PAREN}|[^\s,)\]\n]+)',
        lambda m: f"{_BIG_OP_MAP[m.group(1)]}({m.group(6)}, ({m.group(2).translate(_SUB_MAP)}, {_clean_bound(m.group(3))}, {_clean_bound(m.group(4) or m.group(5))}))",
        code
    )

    # A2. Integrals: ∫[a, b] f dx, ∫_a^b f dx, ∫₀¹ f dx, ∬ f dx dy, ∭ f dx dy dz, ∫ f dx
    DIFF_END = (
        rf'(?:(?<=\s)|(?<=[\)\]{_SUP_CHARS}]))'
        rf'[d∂]({_IDENT})(?![a-zA-Z0-9_πτφϕθ\s]*\()'
    )
    code = re.sub(
        r'[∫∮]\[([^,\]]+),\s*([^\]]+)\]\s*(.+?)' + DIFF_END,
        r'_integrate(\3, (\4, \1, \2))', code
    )
    code = re.sub(
        r'[∫∮]_([^\s\^]+)\^([^\s]+)\s+(.+?)' + DIFF_END,
        lambda m: f"_integrate({m.group(3)}, ({m.group(4)}, {_clean_bound(m.group(1))}, {_clean_bound(m.group(2))}))",
        code
    )
    code = re.sub(
        rf'[∫∮]([{_EXPR_SUB_CHARS}∞]+)(?:\^([a-zA-Z0-9_πτφϕθ∞+\-]+)|([{_SUP_CHARS}πτℯ∞]+))\s*(.+?)' + DIFF_END,
        lambda m: f"_integrate({m.group(4)}, ({m.group(5)}, {_clean_bound(m.group(1))}, {_clean_bound(m.group(2) or m.group(3))}))",
        code
    )
    code = re.sub(
        rf'∭\s*(.+?)\s+[d∂]({_IDENT})\s+[d∂]({_IDENT})\s+[d∂]({_IDENT})\b',
        r'_integrate(\1, \2, \3, \4)', code
    )
    code = re.sub(
        rf'∬\s*(.+?)\s+[d∂]({_IDENT})\s+[d∂]({_IDENT})\b',
        r'_integrate(\1, \2, \3)', code
    )
    code = re.sub(r'[∫∮]\s*(.+?)' + DIFF_END, r'_integrate(\1, \2)', code)

    # B. Derivatives: ∂²/∂x∂y, d²/dx², ∂/∂x, d/dx, ∂ₓ, f′(x), ẋ, ẍ
    code = re.sub(
        rf'[d∂]²/[d∂]({_IDENT}?)[d∂]({_IDENT})',
        r'_DiffOp(\1, \2)', code
    )
    code = re.sub(
        rf'[d∂]([⁰¹²³⁴⁵⁶⁷⁸⁹]+)/[d∂]({_IDENT})\1',
        lambda m: f"_DiffOp({m.group(2)}, {m.group(1).translate(_SUP_MAP)})", code
    )
    code = re.sub(rf'[d∂]/[d∂]({_IDENT})', r'_DiffOp(\1)', code)
    code = re.sub(
        rf'∂([{_IDENT_SUB_CHARS}]+)',
        lambda m: f"_DiffOp({', '.join(m.group(1).translate(_SUB_MAP))})", code
    )
    _PRIME_WEIGHT = {'′': 1, '″': 2, '‴': 3}
    code = re.sub(
        rf'({_IDENT})([′″‴]+)\s*\(',
        lambda m: f"_prime_eval({m.group(1)}, {sum(_PRIME_WEIGHT[c] for c in m.group(2))}, ",
        code
    )
    code = re.sub(
        rf'({_IDENT})([′″‴]+)',
        lambda m: f"_prime_diff({m.group(1)}, {sum(_PRIME_WEIGHT[c] for c in m.group(2))})",
        code
    )
    for ch, base, order in [('ẋ', 'x', 1), ('ẍ', 'x', 2), ('ẏ', 'y', 1), ('ÿ', 'y', 2), ('ż', 'z', 1)]:
        code = re.sub(rf'(?<![a-zA-Z0-9_]){ch}(?![a-zA-Z0-9_])', f'_dot_diff({base}, {order})', code)
    code = re.sub(r'(?<![a-zA-Z0-9_])([a-zA-Z])\u0308(?![a-zA-Z0-9_])', r'_dot_diff(\1, 2)', code)
    code = re.sub(r'(?<![a-zA-Z0-9_])([a-zA-Z])\u0307(?![a-zA-Z0-9_])', r'_dot_diff(\1, 1)', code)

    # C. Vector Calculus: ∇², ∆, ∇⋅, ∇×, ∇
    code = re.sub(r'(?:∇²|∆)\s*(?=\()', '_laplacian', code)
    code = re.sub(rf'(?:∇²|∆)\s*({_IDENT})', r'_laplacian(\1)', code)
    code = re.sub(r'∇\s*[⋅·]\s*(?=\()', '_div', code)
    code = re.sub(rf'∇\s*[⋅·]\s*({_IDENT})', r'_div(\1)', code)
    code = re.sub(r'∇\s*×\s*(?=\()', '_curl', code)
    code = re.sub(rf'∇\s*×\s*({_IDENT})', r'_curl(\1)', code)
    code = re.sub(r'∇\s*(?=\()', '_grad', code)
    code = re.sub(rf'∇\s*({_IDENT})', r'_grad(\1)', code)

    # D. Kronecker Delta (δᵢⱼ), Levi-Civita (εᵢⱼₖ), Combinatorics (ⁿCₖ, ₅C₂, ⁿPₖ, ₅P₂)
    code = re.sub(
        rf'(?<![a-zA-Z0-9_])δ([{_IDENT_SUB_CHARS}]{{2}})(?![{_IDENT_SUB_CHARS}a-zA-Z0-9_])',
        lambda m: f"δ({', '.join(m.group(1).translate(_SUB_MAP))})", code
    )
    code = re.sub(
        rf'(?<![a-zA-Z0-9_])[εϵ]([{_IDENT_SUB_CHARS}]{{2,}})(?![{_IDENT_SUB_CHARS}a-zA-Z0-9_])',
        lambda m: f"ε({', '.join(m.group(1).translate(_SUB_MAP))})", code
    )
    code = re.sub(
        rf'(?<![a-zA-Z0-9_])([{_SUP_CHARS}{_IDENT_SUB_CHARS}]+)C([{_IDENT_SUB_CHARS}]+)(?![a-zA-Z0-9_])',
        lambda m: f"binomial({_clean_bound(m.group(1))}, {_clean_bound(m.group(2))})", code
    )
    code = re.sub(
        rf'(?<![a-zA-Z0-9_])([{_SUP_CHARS}{_IDENT_SUB_CHARS}]+)P([{_IDENT_SUB_CHARS}]+)(?![a-zA-Z0-9_])',
        lambda m: f"permutation({_clean_bound(m.group(1))}, {_clean_bound(m.group(2))})", code
    )

    # E. Brackets, Quantum Bra-Kets, Commutators, Evaluation Bar, Set-Builder & |x|
    while re.search(r'⟦[^⟦⟧]+,[^⟦⟧]+⟧', code):
        code = re.sub(r'⟦([^⟦⟧,]+),\s*([^⟦⟧]+)⟧', r'_commutator(\1, \2)', code)
    code = re.sub(r'\[([^\[\],]+),\s*([^\[\]]+)\]₋', r'_commutator(\1, \2)', code)
    code = re.sub(r'\[([^\[\],]+),\s*([^\[\]]+)\]₊', r'_anticommutator(\1, \2)', code)
    code = re.sub(r'\{([^{},]+),\s*([^{}]+)\}₊', r'_anticommutator(\1, \2)', code)

    while re.search(r'‖[^‖]+‖[₀₁₂₃₄₅₆₇₈₉∞]+', code):
        code = re.sub(
            r'‖([^‖]+)‖([₀₁₂₃₄₅₆₇₈₉∞]+)',
            lambda m: f"_norm({m.group(1)}, {_clean_bound(m.group(2))})", code
        )
    while re.search(r'‖[^‖]+‖', code):
        code = re.sub(r'‖([^‖]+)‖', r'_norm(\1)', code)
    while re.search(r'⌊[^⌊⌋]+⌋', code):
        code = re.sub(r'⌊([^⌊⌋]+)⌋', r'floor(\1)', code)
    while re.search(r'⌈[^⌈⌉]+⌉', code):
        code = re.sub(r'⌈([^⌈⌉]+)⌉', r'ceil(\1)', code)

    # Quantum Bra-Kets: |ψ⟩⟨φ| (outer), ⟨φ|A|ψ⟩ (expectation), ⟨φ|ψ⟩ (inner), |ψ⟩, ⟨φ|
    code = re.sub(
        r'\|([^⟨⟩|]+)⟩\s*⟨([^⟨⟩|]+)\|',
        lambda m: f"(_ket({_clean_qstate(m.group(1))}) * _bra({_clean_qstate(m.group(2))}))",
        code
    )
    while re.search(r'⟨[^⟨⟩|]+\|[^⟨⟩|]+\|[^⟨⟩|]+⟩', code):
        code = re.sub(
            r'⟨([^⟨⟩|]+)\|([^⟨⟩|]+)\|([^⟨⟩|]+)⟩',
            lambda m: f"_expectation({_clean_qstate(m.group(1))}, {m.group(2)}, {_clean_qstate(m.group(3))})",
            code
        )
    while re.search(r'⟨[^⟨⟩|]+\|[^⟨⟩|]+⟩', code):
        code = re.sub(
            r'⟨([^⟨⟩|]+)\|([^⟨⟩|]+)⟩',
            lambda m: f"inner({_clean_qstate(m.group(1))}, {_clean_qstate(m.group(2))})",
            code
        )
    while re.search(r'⟨[^⟨⟩|]+⟩', code):
        code = re.sub(r'⟨([^⟨⟩|]+)⟩', r'inner(\1)', code)
    code = re.sub(r'\|([^⟨⟩|]+)⟩', lambda m: f"_ket({_clean_qstate(m.group(1))})", code)
    code = re.sub(r'⟨([^⟨⟩|]+)\|', lambda m: f"_bra({_clean_qstate(m.group(1))})", code)

    # Definite Integral Evaluation Bar: (x³ / 3)|₀² or (sin(x))|₀^π
    code = re.sub(
        rf'({_BAL_PAREN})\s*[|∣]([{_EXPR_SUB_CHARS}∞]+)(?:\^([a-zA-Z0-9_πτφϕθ∞+\-]+)|([{_SUP_CHARS}πτℯ∞]+))',
        lambda m: f"_eval_bar({m.group(1)}, {_clean_bound(m.group(2))}, {_clean_bound(m.group(3) or m.group(4))})",
        code
    )

    # Set-Builder Notation: {x² ∣ x ∈ [1, 2, 3] : x > 1} or {x² | x ∈ A}
    code = re.sub(
        rf'\{{([^|∣{{}}\n]+?)\s*[|∣]\s*({_IDENT})\s*∈\s*({_BAL_EXPR_NO_COMMA_COLON})(?:\s*[,:]\s*([^{{}}\n]+))?\}}',
        lambda m: f"{{{m.group(1)} for {m.group(2)} in {m.group(3)} if {m.group(4)}}}" if m.group(4) else f"{{{m.group(1)} for {m.group(2)} in {m.group(3)}}}",
        code
    )

    # Absolute Value / Modulus / Determinant / Set Cardinality: |x|, |z|, |M|, |A|
    code = re.sub(
        r'(^\s*|(?<=[=+\-*/(,<>≤≥≠≡≈±∓∈∉∧∨:;\[\{])\s*)\|(?=\S)([^|\n⟨⟩]+?)(?<=\S)\|(?![⟩|])',
        r'\1_abs_det_card(\2)',
        code
    )

    # F. Quantifiers (∀, ∃, ∄), Modular Arithmetic (a ≡ b (mod m)), Limits & Series Bounds
    _QUANT_MAP = {'∀': '_FORALL_', '∃': '_EXISTS_', '∄': '_NEXISTS_'}
    code = re.sub(
        rf'([∀∃∄])\s*({_IDENT})\s*∈\s*({_BAL_EXPR_NO_COMMA_COLON})\s*[,:]\s*({_BAL_EXPR_NO_COMMA})',
        lambda m: f"{_QUANT_MAP[m.group(1)]}({m.group(4)} for {m.group(2)} in {m.group(3)})",
        code
    )
    code = re.sub(
        r'([^=,(\n]+?)\s*≡\s*([^,)\n]+?)\s*\(\s*mod\s+([^)]+)\)',
        r'((((\1) - (\2)) % (\3)) == 0)', code
    )
    code = re.sub(
        r'([^=,(\n]+?)\s*≢\s*([^,)\n]+?)\s*\(\s*mod\s+([^)]+)\)',
        r'((((\1) - (\2)) % (\3)) != 0)', code
    )
    code = re.sub(rf'(\d+|\b{_IDENT})\s*\.\.\s*([^,)\]]+)', r'\1 → \2', code)
    code = re.sub(
        rf'({_IDENT})\s*=\s*([^,→()]+)\s*→\s*([^,)]+)',
        r'(\1, \2, \3)', code
    )
    code = re.sub(
        rf'({_IDENT})\s*→\s*([^,⁺⁻)]+)([⁺⁻])?',
        lambda m: (
            f"{m.group(1)}, {m.group(2)}, {'_DIR_POS_' if m.group(3) == '⁺' else '_DIR_NEG_'}"
            if m.group(3) else f"{m.group(1)}, {m.group(2)}"
        ),
        code
    )

    # G. Subscripts in Variable Names: x₁ -> x_1, aᵢ -> a_i
    code = re.sub(
        rf'([{_IDENT_SUB_CHARS}]+)',
        lambda m: '_' + m.group(1).translate(_SUB_MAP), code
    )

    # H. Postfix Matrix Transpose (Aᵀ), Hermitian Dagger (A†), Overbar Conjugate (z̄)
    code = re.sub(rf'({_IDENT})\u0304', r'conjugate(\1)', code)
    code = re.sub(rf'({_BAL_PAREN}|{_BAL_BRACK}|{_IDENT})\s*ᵀ', r'_transpose(\1)', code)
    code = re.sub(rf'({_BAL_PAREN}|{_BAL_BRACK}|{_IDENT})\s*†', r'_dagger(\1)', code)

    # I. Postfix Degrees (°), Double Factorials (n!!) and Factorials (n!)
    code = re.sub(rf'({_BAL_PAREN})\s*°', r'radians\1', code)
    code = re.sub(r'(\d+(?:\.\d+)?|[a-zA-Z_πτφϕℯ]+)\s*°', r'radians(\1)', code)
    code = re.sub(rf'({_BAL_PAREN})!!(?!=)', r'factorial2\1', code)
    code = re.sub(r'([a-zA-Z0-9_]+)!!(?!=)', r'factorial2(\1)', code)
    code = re.sub(rf'({_BAL_PAREN})!(?!=)', r'factorial\1', code)
    code = re.sub(r'([a-zA-Z0-9_]+)!(?!=)', r'factorial(\1)', code)

    # J0. Trig Powers sin²(x) and Function Inverses sin⁻¹(x), ℒ⁻¹(F), f⁻¹(y)
    code = re.sub(r'\b(sin|cos|tan|sinh|cosh|tanh)⁻¹\s*\(', r'a\1(', code)
    code = re.sub(rf'({_IDENT})⁻¹\s*\(', r'_func_inv(\1)(', code)
    code = re.sub(
        rf'\b(sin|cos|tan|sec|csc|cot|sinh|cosh|tanh|ln|log)([{_SUP_CHARS}]+)\s*({_BAL_PAREN})',
        r'(\1\3)\2', code
    )

    # J1. Superscript Exponents: x² -> x**(2), ℯ⁻ˣ² -> ℯ**(-x**2)
    code = re.sub(
        rf'([{_SUP_CHARS}]+)',
        lambda m: f"**({_translate_sup(m.group(1))})", code
    )

    # K. Prefix Roots: √x, ∛x, ∜x, √(x+1)
    code = re.sub(r'(\d|[a-zA-Z_πτφϕℯ𝕚ⅈℏ\)\]])\s*([√∛∜])', r'\1*\2', code)
    code = re.sub(r'√\s*\(', 'sqrt(', code)
    code = re.sub(r'∛\s*\(', 'cbrt(', code)
    code = re.sub(r'∜\s*\(', 'fourth_root(', code)
    code = re.sub(r'√\s*([a-zA-Z0-9_πτφϕℯ𝕚ⅈℏ]+(?:\*\*\([^)]+\))?)', r'sqrt(\1)', code)
    code = re.sub(r'∛\s*([a-zA-Z0-9_πτφϕℯ𝕚ⅈℏ]+(?:\*\*\([^)]+\))?)', r'cbrt(\1)', code)
    code = re.sub(r'∜\s*([a-zA-Z0-9_πτφϕℯ𝕚ⅈℏ]+(?:\*\*\([^)]+\))?)', r'fourth_root(\1)', code)

    # L. Implicit Multiplication: 2x, 3𝕚, 2π, 2(x+1), (x+1)(x-1), x²y
    def _num_ident_mult(m):
        num, ident, rest = m.group(1), m.group(2), m.string[m.end():]
        if num == '0' and ident[0] in 'xXbBoO':
            return m.group(0)
        if ident[0] in 'eE' and (
            (len(ident) == 1 and rest.startswith(('+', '-'))) or ident[1:].isdigit()
        ):
            return m.group(0)
        return f"{num}*{ident}"

    code = re.sub(r'(?<![a-zA-Z0-9_\.])(\d+(?:\.\d+)?)([a-zA-Z_πτφϕℯ𝕚ⅈℏ∑∏ΣΠ]+)', _num_ident_mult, code)
    code = re.sub(r'(?<![a-zA-Z0-9_])(\d+(?:\.\d+)?)\s*\(', r'\1*(', code)
    code = re.sub(rf'(?<![a-zA-Z0-9_\)\]])({_BAL_PAREN})\s*\(', r'\1*(', code)
    code = re.sub(r'\)([a-zA-Z0-9_πτφϕℯ𝕚ⅈℏ])', r')*\1', code)
    code = re.sub(r'(?<![a-zA-Z0-9_])([πτφϕℯ𝕚ⅈℏ])\s*([a-zA-Zπτφϕℯ𝕚ⅈℏ\(])', r'\1*\2', code)

    # M. Lambda Functions: x ↦ 2x + 1
    code = re.sub(r'\(([^()]+)\)\s*↦\s*', r'lambda \1: ', code)
    code = re.sub(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*↦\s*', r'lambda \1: ', code)

    # N. Infix Operators: ×, ⊗, ⊕, ⋀/∧, ⊛, ⊡/∶, ±, ∓, ≈, ≟, ∼, ∝, ⟹, ⟺, ⊥, ∥, ∣, ∤
    code = re.sub(r'×', ' @ _CROSS @ ', code)
    code = re.sub(r'⊗', ' @ _TENSOR @ ', code)
    code = re.sub(r'⊕', ' @ _OPLUS @ ', code)
    code = re.sub(r'⋀', ' @ _WEDGE @ ', code)
    code = re.sub(r'⊛', ' @ _CONVOLVE @ ', code)
    code = re.sub(r'[⊡∶]', ' @ _DCONTRACT @ ', code)
    if not re.search(r'(?:[<>!]=?|==|\bnot\b|\bin\b|[≤⩽≥⩾≠≢≡∈∉≈∣∤⊆⊂⊇⊃¬])', code):
        code = re.sub(r'∧', ' @ _WEDGE @ ', code)

    code = re.sub(r'(?<=[a-zA-Z0-9_πτφϕℯ𝕚ⅈℏ\)\]])\s*±\s*', ' << _PM >> ', code)
    code = re.sub(r'±\s*', '_PM >> ', code)
    code = re.sub(r'(?<=[a-zA-Z0-9_πτφϕℯ𝕚ⅈℏ\)\]])\s*∓\s*', ' << _MP >> ', code)
    code = re.sub(r'∓\s*', '_MP >> ', code)
    code = re.sub(r'≈', ' << _APPROX >> ', code)
    code = re.sub(r'≟', ' << _SYMEQ >> ', code)
    code = re.sub(r'∼', ' << _ASYMP >> ', code)
    code = re.sub(r'∝', ' << _PROPTO >> ', code)
    code = re.sub(r'⟹', ' << _IMPLIES >> ', code)
    code = re.sub(r'⟺', ' << _IFF >> ', code)
    code = re.sub(r'⊥', ' << _ORTHO >> ', code)
    code = re.sub(r'∥', ' << _PARALLEL >> ', code)
    code = re.sub(r'∣', ' << _DIVIDES >> ', code)
    code = re.sub(r'∤', ' << _NDIVIDES >> ', code)

    # O. Direct Symbol & Constant Replacements
    for sym, py_op in [
        ('^', '**'),
        ('⋅', '*'), ('·', '*'), ('÷', '/'), ('−', '-'),
        ('≤', '<='), ('⩽', '<='), ('≥', '>='), ('⩾', '>='), ('≠', '!='), ('≢', '!='), ('≡', '=='),
        ('∉', ' not in '), ('∈', ' in '), ('¬', ' not '), ('∧', ' and '), ('∨', ' or '),
        ('∪', '|'), ('∩', '&'), ('∖', '-'), ('△', '^'), ('⊆', '<='), ('⊂', '<'), ('⊇', '>='), ('⊃', '>'),
        ('∑', 'Σ'), ('∏', 'Π'), ('⋃', '_BIGUNION_'), ('⋂', '_BIGINTER_'),
        ('∀', '_FORALL_'), ('∃', '_EXISTS_'), ('∄', '_NEXISTS_'),
        ('∞', '_OO_'), ('∅', '_EMPTYSET_'), ('∂', '_PARTIAL_'),
        ('ℯ', '_E_'), ('𝕚', '_I_'), ('ⅈ', '_I_'),
        ('ℒ', '_LAPLACE_'), ('ℱ', '_FOURIER_'),
        ('ℕ', '_NATURALS_'), ('ℤ', '_INTEGERS_'), ('ℚ', '_RATIONALS_'),
        ('ℝ', '_REALS_'), ('ℂ', '_COMPLEXES_'), ('ℜ', '_RE_'), ('ℑ', '_IM_')
    ]:
        code = code.replace(sym, py_op)

    return code

def _transform_code_segment(segment):
    return "\n".join(_transform_code_line(line) for line in segment.split("\n"))

def math_syntax_transformer(lines):
    full_text = "".join(lines)
    if "_MATH_SYNTAX_SETUP_GUARD_" in full_text:
        return lines
    parts = _TOKEN_SPLIT.split(full_text)
    parts[0::2] = [_transform_code_segment(p) for p in parts[0::2]]
    return "".join(parts).splitlines(keepends=True)

_EXPORTED_NAMES = {
    k_: v_ for k_, v_ in globals().items()
    if not k_.startswith('__') and k_ not in ('re', 'math', 'cmath', 'operator', 'itertools')
}

def load_ipython_extension(ipython):
    """Register the math syntax transformer and inject math symbols/helpers into the notebook namespace."""
    ipython.user_ns.update(_EXPORTED_NAMES)
    ipython.input_transformers_cleanup[:] = [
        f for f in ipython.input_transformers_cleanup
        if getattr(f, '__name__', '') != 'math_syntax_transformer'
    ]
    ipython.input_transformers_cleanup.append(math_syntax_transformer)
    print("Full textbook math syntax enabled!")

def unload_ipython_extension(ipython):
    """Remove the math syntax transformer from IPython."""
    ipython.input_transformers_cleanup[:] = [
        f for f in ipython.input_transformers_cleanup
        if getattr(f, '__name__', '') != 'math_syntax_transformer'
    ]
    print("Math syntax transformer disabled.")
