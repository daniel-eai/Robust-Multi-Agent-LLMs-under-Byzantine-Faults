#!/usr/bin/env python3
"""math_grader.py — MATH-500 answer extraction + equivalence checking.

Self-contained (sympy only, no antlr4). Used on the math500 path; set
MATH_BOXED_GRADING=0 to fall back to plain numeric parsing. Implements the standard MATH normalization (Hendrycks/Minerva style)
plus a sympy symbolic-equivalence fallback so answers like \\frac{14}{3}, 3\\sqrt{13},
(3,\\frac{\\pi}{2}), 90^\\circ, \\text{Evelyn} grade correctly.
"""
import re

_BOXED_RE = re.compile(r"\\boxed\s*")


def last_boxed(text: str):
    """Return the content of the LAST \\boxed{...} (brace-balanced), or None."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i = idx
    while i < len(text) and text[i] != "{":
        i += 1
    if i >= len(text):
        return None
    depth = 0
    start = i
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:j]
    return None


def _clean_extracted(s: str) -> str:
    """Strip 'Answer:' prefixes, $ delimiters, and whitespace from an answer."""
    if s is None:
        return ""
    s = s.strip()
    # peel repeated wrappers: $...$, 'Answer:' / 'Final answer:' prefixes
    for _ in range(4):
        s2 = s
        s2 = re.sub(r"^\**\s*(?:the\s+)?(?:final\s+)?answer\s*(?:is)?\s*[:=]?\s*",
                    "", s2, flags=re.IGNORECASE).strip()
        s2 = s2.strip("$").strip()
        s2 = s2.strip("*").strip()
        if s2 == s:
            break
        s = s2
    return s


def extract_answer(text: str) -> str:
    """Pull the final answer string from a model response. Priority:
    \\boxed{...} > 'Answer: ...' line > last non-empty $...$ > last number."""
    if not text:
        return ""
    # Drop Qwen3 <think>...</think> reasoning so the answer is read from the final
    # part (thinking may contain intermediate \boxed{} steps and stray numbers).
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    else:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    b = last_boxed(text)
    if b is not None:
        c = _clean_extracted(b)
        if c:
            return c
    m = re.search(r"(?:final answer|answer)\s*[:=]\s*(.+?)\s*$",
                  text, re.IGNORECASE | re.MULTILINE)
    if m:
        c = _clean_extracted(m.group(1).rstrip("."))
        if c:
            return c
    spans = [s for s in re.findall(r"\$([^$]+?)\$", text) if s.strip()]
    if spans:
        return _clean_extracted(spans[-1])
    nums = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", text)
    return nums[-1] if nums else text.strip()[:64]


def _strip(s: str) -> str:
    """Normalize a LaTeX/plain answer string (MATH-eval style)."""
    if s is None:
        return ""
    s = str(s).strip()
    b = last_boxed(s)
    if b is not None:
        s = b
    # peel 'Answer:' prefixes and $ wrappers
    for _ in range(3):
        s2 = re.sub(r"^\**\s*(?:the\s+)?(?:final\s+)?answer\s*(?:is)?\s*[:=]?\s*",
                    "", s, flags=re.IGNORECASE).strip().strip("$").strip("*").strip()
        if s2 == s:
            break
        s = s2
    # strip common trailing units/words
    s = re.sub(r"\b(cm|mm|km|meters?|centimeters?|inches|inch|feet|ft|units?|"
               r"degrees?|radians?|dollars?|cents?|square\s*\w+|cubic\s*\w+)\b\.?\s*$",
               "", s, flags=re.IGNORECASE).strip()
    # remove wrappers / formatting
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\ ", " ")
    s = s.replace("$", "").replace("\\$", "")
    s = re.sub(r"\\text\s*\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mbox\s*\{([^}]*)\}", r"\1", s)
    s = s.replace("^\\circ", "").replace("^{\\circ}", "").replace("\\circ", "")
    s = s.replace("\\%", "").replace("\\$", "").replace("%", "")
    s = s.replace("dollars", "").replace("dollar", "")
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("{,}", "").replace(",", "") if re.search(r"\d,\d{3}", s) else s
    s = s.replace("\\%", "").strip()
    s = s.rstrip(".")
    s = s.replace(" ", "")
    # units like \pi kept; degree removed above
    return s.strip().lower()


def _to_sympy_str(s: str) -> str:
    """Convert normalized LaTeX to a sympy-parseable expression string."""
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    # \frac{a}{b} -> ((a)/(b))  (repeat for nesting)
    for _ in range(6):
        s2 = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", s)
        if s2 == s:
            break
        s = s2
    # a/b style already fine
    s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\sqrt(\w)", r"sqrt(\1)", s)
    s = s.replace("\\pi", "pi")
    s = s.replace("^", "**")
    s = re.sub(r"\\[a-zA-Z]+", "", s)  # drop leftover latex commands
    s = s.replace("{", "(").replace("}", ")")
    # implicit multiplication: 3sqrt -> 3*sqrt, 2pi -> 2*pi, )( -> )*(
    s = re.sub(r"(\d)([a-zA-Z(])", r"\1*\2", s)
    s = re.sub(r"([a-zA-Z0-9)])\(", r"\1*(", s)  # careful: keeps sqrt( intact? sqrt( -> sqrt*( bad
    s = s.replace("sqrt*(", "sqrt(").replace("pi*(", "pi*(")
    return s


def _sympy_eq(a: str, b: str) -> bool:
    try:
        import sympy
        from sympy import simplify, sympify, nsimplify, N
        ea = sympify(_to_sympy_str(a), rational=True)
        eb = sympify(_to_sympy_str(b), rational=True)
        if ea == eb:
            return True
        d = simplify(ea - eb)
        if d == 0:
            return True
        return abs(float(N(ea)) - float(N(eb))) < 1e-6
    except Exception:
        return False


def _split_tuple(s: str):
    s = s.strip()
    if len(s) >= 2 and s[0] in "([" and s[-1] in ")]":
        inner = s[1:-1]
        # split on top-level commas
        parts, depth, cur = [], 0, ""
        for ch in inner:
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            if ch == "," and depth == 0:
                parts.append(cur); cur = ""
            else:
                cur += ch
        parts.append(cur)
        return [p.strip() for p in parts]
    return None


def math_equal(pred: str, gt: str) -> bool:
    """True if pred is mathematically equivalent to gt."""
    if pred is None or gt is None:
        return False
    ps, gs = _strip(pred), _strip(gt)
    if ps == gs and ps != "":
        return True
    # tuple / interval element-wise
    pt, gtp = _split_tuple(ps), _split_tuple(gs)
    if pt is not None and gtp is not None and len(pt) == len(gtp):
        return all(math_equal(a, b) for a, b in zip(pt, gtp))
    if (pt is None) != (gtp is None):
        return False
    return _sympy_eq(ps, gs)


if __name__ == "__main__":
    tests = [
        (r"\frac{14}{3}", "14/3", True),
        (r"4.666", "14/3", False),
        (r"3\sqrt{13}", "3*sqrt(13)", True),
        (r"\left( 3, \frac{\pi}{2} \right)", "(3,pi/2)", True),
        (r"90^\circ", "90", True),
        (r"\text{Evelyn}", "Evelyn", True),
        (r"\frac{3}{56}", "3/56", True),
        ("2", "2", True),
        ("2", "3", False),
        (r"\sqrt{51}", "sqrt(51)", True),
    ]
    for p, g, exp in tests:
        got = math_equal(p, g)
        print(("OK " if got == exp else "XX ") + f"{p!r} vs {g!r} -> {got} (exp {exp})")
