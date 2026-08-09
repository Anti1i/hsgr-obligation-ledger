"""Pragmatic MATH answer normalization / equivalence checking.

Not a full math-verify replacement (py3.9 server, no antlr), but consistent
across all experimental conditions so comparisons stay fair.
"""
import re
from fractions import Fraction


def extract_boxed(text):
    """Extract content of the last \\boxed{...} (brace-balanced)."""
    idx = text.rfind("\\boxed")
    if idx == -1:
        m = re.findall(r"(?:answer is|Answer:)\s*([^\n.]+)", text)
        return m[-1].strip() if m else None
    i = text.find("{", idx)
    if i == -1:
        return None
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j]
        j += 1
    return None


def normalize_answer(ans):
    if ans is None:
        return None
    s = ans.strip()
    s = re.sub(r"\\(text|mbox|textbf|mathrm)\s*\{([^{}]*)\}", r"\2", s)
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\ ", "")
    s = s.replace("dfrac", "frac").replace("tfrac", "frac")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = s.replace("\\%", "").replace("%", "")
    s = s.replace("\\$", "").replace("$", "")
    s = re.sub(r"\s+", "", s)
    # \frac{a}{b} -> a/b for simple numeric fractions
    m = re.fullmatch(r"\\frac\{(-?\d+)\}\{(\d+)\}", s)
    if m:
        s = m.group(1) + "/" + m.group(2)
    m = re.fullmatch(r"(-?\d+)\\frac\{(\d+)\}\{(\d+)\}", s)  # mixed number
    if m:
        whole, num, den = int(m.group(1)), int(m.group(2)), int(m.group(3))
        sign = -1 if whole < 0 else 1
        s = str(Fraction(whole) + sign * Fraction(num, den))
    s = s.rstrip(".")
    s = re.sub(r",(?=\d{3}\b)", "", s)  # thousands separators
    return s


def _to_number(s):
    try:
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        pass
    m = re.fullmatch(r"(-?\d+)/(\d+)", s)
    if m:
        try:
            return Fraction(int(m.group(1)), int(m.group(2)))
        except ZeroDivisionError:
            return None
    try:
        f = float(s)
        return Fraction(f).limit_denominator(10**9)
    except (ValueError, OverflowError):
        return None


def _sympy_equal(a, b):
    try:
        import sympy
        from sympy.parsing.sympy_parser import (
            parse_expr,
            standard_transformations,
            implicit_multiplication_application,
        )

        tr = standard_transformations + (implicit_multiplication_application,)

        def prep(x):
            x = x.replace("\\pi", "pi").replace("\\sqrt", "sqrt")
            x = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", x)
            x = re.sub(r"sqrt\{([^{}]*)\}", r"sqrt(\1)", x)
            x = x.replace("^", "**").replace("{", "(").replace("}", ")")
            return x

        ea = parse_expr(prep(a), transformations=tr)
        eb = parse_expr(prep(b), transformations=tr)
        return bool(sympy.simplify(ea - eb) == 0)
    except Exception:
        return False


def answers_equal(pred, gold):
    p, g = normalize_answer(pred), normalize_answer(gold)
    if p is None or g is None:
        return False
    if p == g:
        return True
    if p.lower() == g.lower():
        return True
    np_, ng = _to_number(p), _to_number(g)
    if np_ is not None and ng is not None:
        return np_ == ng
    if (np_ is None) != (ng is None):
        return False
    return _sympy_equal(p, g)
