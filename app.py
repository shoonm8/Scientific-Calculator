"""
Scientific Calculator — Streamlit App
======================================
A real, working scientific calculator built with Python + Streamlit.

IMPORTANT SAFETY NOTE:
This app does NOT use Python's eval(). Instead, it includes a small,
hand-written, safe expression parser (a "recursive descent parser") near
the top of the file. It only understands numbers, +, -, *, /, ^, !, %,
parentheses, and a fixed list of math functions/constants — nothing else.
It can never run arbitrary Python code, so it is safe to use with any
text the user types or builds using the on-screen buttons.

Run with:
    pip install streamlit
    streamlit run app.py
"""

import math
import streamlit as st

# ============================================================
# 1. SAFE MATH EXPRESSION ENGINE (tokenizer + parser + evaluator)
# ============================================================
# This section turns a text expression like "2+3*4" or "sin(30)" into a
# number, WITHOUT using eval(). It works in three steps:
#   1) tokenize()  -> breaks the string into a list of tokens
#   2) Parser class -> reads the tokens using grammar rules (recursive
#      descent) and computes the answer directly as it reads
#   3) evaluate_expression() -> the single entry point used by the app


def tokenize(expr: str):
    """Turn a string like '2+3*sin(30)' into a list of (TYPE, VALUE) tokens."""
    tokens = []
    i = 0
    n = len(expr)
    while i < n:
        c = expr[i]

        if c.isspace():
            i += 1
            continue

        # Numbers: digits, decimal point, and optional scientific
        # notation using a capital 'E' (e.g. 5E3 means 5 * 10^3).
        # NOTE: lowercase 'e' is reserved for Euler's number, so there
        # is no confusion between "1E3" (scientific notation) and "e"
        # (the constant 2.71828...).
        if c.isdigit() or c == '.':
            j = i
            while j < n and (expr[j].isdigit() or expr[j] == '.'):
                j += 1
            if j < n and expr[j] == 'E':
                k = j + 1
                if k < n and expr[k] in '+-':
                    k += 1
                start_digits = k
                while k < n and expr[k].isdigit():
                    k += 1
                if k > start_digits:  # there really were exponent digits
                    tokens.append(('NUMBER', expr[i:k]))
                    i = k
                    continue
            tokens.append(('NUMBER', expr[i:j]))
            i = j
            continue

        # Identifiers: function names and constants (sin, cos, log, pi, e, Ans, ...)
        if c.isalpha() or c == '_':
            j = i
            while j < n and (expr[j].isalnum() or expr[j] == '_'):
                j += 1
            tokens.append(('IDENT', expr[i:j]))
            i = j
            continue

        # Single-character operators / punctuation
        if c in '+-*/^!%()':
            tokens.append(('OP', c))
            i += 1
            continue

        raise ValueError(f"Invalid character in expression: '{c}'")

    return tokens


class Parser:
    """
    Recursive-descent parser/evaluator for arithmetic expressions.

    Grammar (from lowest to highest precedence):
        expression := term (('+' | '-') term)*
        term       := unary (('*' | '/' | implicit-multiply) unary)*
        unary      := '-' unary | '+' unary | power
        power      := postfix ('^' unary)?         (right-associative)
        postfix    := primary ('!' | '%')*
        primary    := NUMBER | IDENT | IDENT '(' expression ')' | '(' expression ')'
    """

    def __init__(self, tokens, angle_mode, ans_value):
        self.tokens = tokens
        self.pos = 0
        self.angle_mode = angle_mode   # 'deg' or 'rad'
        self.ans_value = ans_value     # value of the last answer ("Ans")

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self):
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect_op(self, char):
        tok = self.peek()
        if tok is None or tok[0] != 'OP' or tok[1] != char:
            raise ValueError(f"Expected '{char}'")
        self.consume()

    # ---- grammar rules, from lowest to highest precedence ----

    def parse_expression(self):
        value = self.parse_term()
        while True:
            tok = self.peek()
            if tok and tok[0] == 'OP' and tok[1] in ('+', '-'):
                self.consume()
                right = self.parse_term()
                value = value + right if tok[1] == '+' else value - right
            else:
                break
        return value

    def parse_term(self):
        value = self.parse_unary()
        while True:
            tok = self.peek()
            if tok and tok[0] == 'OP' and tok[1] == '*':
                self.consume()
                value = value * self.parse_unary()
            elif tok and tok[0] == 'OP' and tok[1] == '/':
                self.consume()
                divisor = self.parse_unary()
                if divisor == 0:
                    raise ValueError("Division by zero")
                value = value / divisor
            elif tok and (tok[0] in ('NUMBER', 'IDENT') or (tok[0] == 'OP' and tok[1] == '(')):
                # Implicit multiplication, e.g. "2pi" or "2(3+4)"
                value = value * self.parse_unary()
            else:
                break
        return value

    def parse_unary(self):
        tok = self.peek()
        if tok and tok[0] == 'OP' and tok[1] == '-':
            self.consume()
            return -self.parse_unary()
        if tok and tok[0] == 'OP' and tok[1] == '+':
            self.consume()
            return self.parse_unary()
        return self.parse_power()

    def parse_power(self):
        base = self.parse_postfix()
        tok = self.peek()
        if tok and tok[0] == 'OP' and tok[1] == '^':
            self.consume()
            exponent = self.parse_unary()  # right-associative, allows 2^-3
            return base ** exponent
        return base

    def parse_postfix(self):
        value = self.parse_primary()
        while True:
            tok = self.peek()
            if tok and tok[0] == 'OP' and tok[1] == '!':
                self.consume()
                if value < 0 or value != int(value):
                    raise ValueError("Factorial needs a non-negative whole number")
                value = math.factorial(int(value))
            elif tok and tok[0] == 'OP' and tok[1] == '%':
                self.consume()
                value = value / 100
            else:
                break
        return value

    def parse_primary(self):
        tok = self.peek()
        if tok is None:
            raise ValueError("Unexpected end of expression")

        if tok[0] == 'NUMBER':
            self.consume()
            return float(tok[1])

        if tok[0] == 'IDENT':
            name = tok[1]
            self.consume()
            nxt = self.peek()
            if nxt and nxt[0] == 'OP' and nxt[1] == '(':
                self.consume()
                arg = self.parse_expression()
                self.expect_op(')')
                return self.call_function(name, arg)
            if name == 'pi':
                return math.pi
            if name == 'e':
                return math.e
            if name == 'Ans':
                return self.ans_value
            raise ValueError(f"Unknown name: {name}")

        if tok[0] == 'OP' and tok[1] == '(':
            self.consume()
            value = self.parse_expression()
            self.expect_op(')')
            return value

        raise ValueError(f"Unexpected token: {tok[1]}")

    def call_function(self, name, arg):
        deg = self.angle_mode == 'deg'
        if name == 'sin':
            return math.sin(math.radians(arg) if deg else arg)
        if name == 'cos':
            return math.cos(math.radians(arg) if deg else arg)
        if name == 'tan':
            return math.tan(math.radians(arg) if deg else arg)
        if name == 'asin':
            r = math.asin(arg)
            return math.degrees(r) if deg else r
        if name == 'acos':
            r = math.acos(arg)
            return math.degrees(r) if deg else r
        if name == 'atan':
            r = math.atan(arg)
            return math.degrees(r) if deg else r
        if name == 'log':
            return math.log10(arg)
        if name == 'ln':
            return math.log(arg)
        if name == 'sqrt':
            if arg < 0:
                raise ValueError("Cannot take the square root of a negative number")
            return math.sqrt(arg)
        raise ValueError(f"Unknown function: {name}")


def evaluate_expression(expr: str, angle_mode: str, ans_value: float):
    """Safely evaluate a math expression string and return a float."""
    if not expr or not expr.strip():
        return 0.0
    tokens = tokenize(expr)
    parser = Parser(tokens, angle_mode, ans_value)
    result = parser.parse_expression()
    if parser.pos != len(parser.tokens):
        raise ValueError("Unexpected characters at the end of the expression")
    return result


def format_number(value: float) -> str:
    """Turn a float into a clean display string (no ugly floating point tails)."""
    value = round(value, 10)
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return str(value)


# ============================================================
# 2. STREAMLIT PAGE SETUP
# ============================================================
st.set_page_config(page_title="Scientific Calculator", page_icon="🧮", layout="centered")

# ---- Session state (the calculator's "memory" between button clicks) ----
if "expr" not in st.session_state:
    st.session_state.expr = ""          # the text currently shown/typed
if "angle_mode" not in st.session_state:
    st.session_state.angle_mode = "deg"  # "deg" or "rad"
if "inv" not in st.session_state:
    st.session_state.inv = False         # inverse trig mode on/off
if "ans" not in st.session_state:
    st.session_state.ans = 0.0           # last computed answer
if "just_evaluated" not in st.session_state:
    st.session_state.just_evaluated = False  # True right after pressing "="

# ============================================================
# 3. BUTTON CALLBACK FUNCTIONS
# ============================================================

def press(token: str):
    """Called when a normal button (digit / operator / function) is pressed."""
    # If the last action was "=" and the user now types a new number,
    # start a fresh expression instead of appending to the old result.
    if st.session_state.just_evaluated and (token.isdigit() or token == "."):
        st.session_state.expr = ""
    # If the display shows an error, any new key clears it first.
    if st.session_state.expr == "Error":
        st.session_state.expr = ""
    st.session_state.expr += token
    st.session_state.just_evaluated = False


def press_function(token: str):
    """Called for sin/cos/tan, respecting the Inv toggle (-> asin/acos/atan)."""
    base_name = token[:-1]  # strip trailing "("
    if st.session_state.inv and base_name in ("sin", "cos", "tan"):
        token = "a" + token
    press(token)
    st.session_state.inv = False  # Inv applies to the next trig press only


def toggle_inv():
    st.session_state.inv = not st.session_state.inv


def set_angle_mode(mode: str):
    st.session_state.angle_mode = mode


def clear_all():
    st.session_state.expr = ""
    st.session_state.just_evaluated = False


def calculate():
    try:
        result = evaluate_expression(
            st.session_state.expr, st.session_state.angle_mode, st.session_state.ans
        )
        st.session_state.ans = result
        st.session_state.expr = format_number(result)
    except Exception:
        st.session_state.expr = "Error"
    st.session_state.just_evaluated = True


# ============================================================
# 4. CUSTOM CSS — makes it look like a clean, modern calculator
# ============================================================
st.markdown(
    """
    <style>
    .block-container {
        max-width: 460px;
        padding-top: 2rem;
        margin: auto;
    }

    /* Tighten the gap between rows of buttons */
    div[data-testid="stVerticalBlock"] {
        gap: 6px !important;
    }
    /* Tighten the gap between columns (buttons within a row) */
    div[data-testid="stHorizontalBlock"] {
        gap: 6px !important;
    }
    div[data-testid="column"] {
        padding: 0px !important;
    }
    div[data-testid="element-container"] {
        margin-bottom: 0px !important;
    }

    .calc-title {
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
        color: #202124;
        margin-bottom: 0.6rem;
    }

    .calc-display {
        background-color: #f1f3f4;
        border-radius: 14px;
        padding: 18px 16px;
        margin-bottom: 12px;
        text-align: right;
        font-size: 2rem;
        font-weight: 600;
        color: #202124;
        min-height: 56px;
        overflow-x: auto;
        white-space: nowrap;
        border: 1px solid #e0e0e0;
    }

    .mode-badge {
        text-align: right;
        color: #1a73e8;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 4px;
        letter-spacing: 0.5px;
    }

    /* Base button style: pill-shaped, tightly packed, equal size */
    .stButton > button {
        width: 100%;
        height: 46px;
        border-radius: 999px;
        border: 1px solid #e3e3e3;
        background-color: #f8f9fa;
        color: #3c4043;
        font-size: 0.95rem;
        font-weight: 600;
        padding: 0px;
        margin: 0px;
        transition: background-color 0.15s ease;
    }
    .stButton > button:hover {
        background-color: #e8eaed;
        border-color: #d2d2d2;
        color: #202124;
    }

    /* Primary buttons ("=" and the active Deg/Rad/Inv toggle) stand out in blue */
    .stButton > button[kind="primary"] {
        background-color: #1a73e8;
        color: #ffffff;
        border: none;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #1765cc;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 5. MAIN CALCULATOR UI
# ============================================================
with st.container(border=True):
    st.markdown('<div class="calc-title">Scientific Calculator</div>', unsafe_allow_html=True)

    # Small badge showing the current angle mode, so it is always visible
    mode_label = "DEG" if st.session_state.angle_mode == "deg" else "RAD"
    inv_label = " · INV" if st.session_state.inv else ""
    st.markdown(
        f'<div class="mode-badge">{mode_label}{inv_label}</div>', unsafe_allow_html=True
    )

    # The display box: shows the expression, right-aligned
    display_text = st.session_state.expr if st.session_state.expr != "" else "0"
    st.markdown(f'<div class="calc-display">{display_text}</div>', unsafe_allow_html=True)

    # ---- Keypad: 7 columns x 5 rows, in the exact order requested ----

    def is_active(mode_name):
        return "primary" if st.session_state.angle_mode == mode_name else "secondary"

    def inv_type():
        return "primary" if st.session_state.inv else "secondary"

    # Row 1: Deg | Rad | x! | ( | ) | % | AC
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    with c1:
        st.button("Deg", key="btn_deg", type=is_active("deg"),
                  on_click=set_angle_mode, args=("deg",), use_container_width=True)
    with c2:
        st.button("Rad", key="btn_rad", type=is_active("rad"),
                  on_click=set_angle_mode, args=("rad",), use_container_width=True)
    with c3:
        st.button("x!", key="btn_fact", on_click=press, args=("!",), use_container_width=True)
    with c4:
        st.button("(", key="btn_lparen", on_click=press, args=("(",), use_container_width=True)
    with c5:
        st.button(")", key="btn_rparen", on_click=press, args=(")",), use_container_width=True)
    with c6:
        st.button("%", key="btn_pct", on_click=press, args=("%",), use_container_width=True)
    with c7:
        st.button("AC", key="btn_ac", on_click=clear_all, use_container_width=True)

    # Row 2: Inv | sin | ln | 7 | 8 | 9 | ÷
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    with c1:
        st.button("Inv", key="btn_inv", type=inv_type(),
                   on_click=toggle_inv, use_container_width=True)
    with c2:
        st.button("sin", key="btn_sin", on_click=press_function, args=("sin(",), use_container_width=True)
    with c3:
        st.button("ln", key="btn_ln", on_click=press, args=("ln(",), use_container_width=True)
    with c4:
        st.button("7", key="btn_7", on_click=press, args=("7",), use_container_width=True)
    with c5:
        st.button("8", key="btn_8", on_click=press, args=("8",), use_container_width=True)
    with c6:
        st.button("9", key="btn_9", on_click=press, args=("9",), use_container_width=True)
    with c7:
        st.button("÷", key="btn_div", on_click=press, args=("/",), use_container_width=True)

    # Row 3: π | cos | log | 4 | 5 | 6 | ×
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    with c1:
        st.button("π", key="btn_pi", on_click=press, args=("pi",), use_container_width=True)
    with c2:
        st.button("cos", key="btn_cos", on_click=press_function, args=("cos(",), use_container_width=True)
    with c3:
        st.button("log", key="btn_log", on_click=press, args=("log(",), use_container_width=True)
    with c4:
        st.button("4", key="btn_4", on_click=press, args=("4",), use_container_width=True)
    with c5:
        st.button("5", key="btn_5", on_click=press, args=("5",), use_container_width=True)
    with c6:
        st.button("6", key="btn_6", on_click=press, args=("6",), use_container_width=True)
    with c7:
        st.button("×", key="btn_mul", on_click=press, args=("*",), use_container_width=True)

    # Row 4: e | tan | √ | 1 | 2 | 3 | −
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    with c1:
        st.button("e", key="btn_e", on_click=press, args=("e",), use_container_width=True)
    with c2:
        st.button("tan", key="btn_tan", on_click=press_function, args=("tan(",), use_container_width=True)
    with c3:
        st.button("√", key="btn_sqrt", on_click=press, args=("sqrt(",), use_container_width=True)
    with c4:
        st.button("1", key="btn_1", on_click=press, args=("1",), use_container_width=True)
    with c5:
        st.button("2", key="btn_2", on_click=press, args=("2",), use_container_width=True)
    with c6:
        st.button("3", key="btn_3", on_click=press, args=("3",), use_container_width=True)
    with c7:
        st.button("−", key="btn_sub", on_click=press, args=("-",), use_container_width=True)

    # Row 5: Ans | EXP | xʸ | 0 | . | = | +
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    with c1:
        st.button("Ans", key="btn_ans", on_click=press, args=("Ans",), use_container_width=True)
    with c2:
        st.button("EXP", key="btn_exp", on_click=press, args=("E",), use_container_width=True)
    with c3:
        st.button("xʸ", key="btn_pow", on_click=press, args=("^",), use_container_width=True)
    with c4:
        st.button("0", key="btn_0", on_click=press, args=("0",), use_container_width=True)
    with c5:
        st.button(".", key="btn_dot", on_click=press, args=(".",), use_container_width=True)
    with c6:
        st.button("=", key="btn_eq", type="primary", on_click=calculate, use_container_width=True)
    with c7:
        st.button("+", key="btn_add", on_click=press, args=("+",), use_container_width=True)
