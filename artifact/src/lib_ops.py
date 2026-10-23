# -*- coding: utf-8 -*-
"""
lib_ops.py —— C2 核心：单元布尔功能 → "库无关"原语算子计数（真值表模板匹配）
============================================================================

动机（C2 设计点）：跨工艺库粒度/写法不一致。同一 AOI21 功能，三库三种写法：
    Nangate :  !(A | (B1 & B2))          （因式分解+输出取反）
    sky130  :  (!A1&!B1) | (!A2&!B1)     （DeMorgan 展开）
    asap7   :  (!A1 * !B) + (!A2 * !B)   （SOP 展开）
按表达式数算子跨库不等价（实测 and:1 vs and:2 vs and:2）；而纯 SOP 规范化
"功能不变但分解不加"（AOI21 格子 7 ops ≠ 拆开的 and+or+inv 3 ops）。

解法：**真值表 + 因式分解模板匹配**，要求分解可加性：
    count(AOI21 格子) == count(and 格)+count(or 格)+count(inv 格) == {and:1,or:1,not:1}
  1. function 表达式（任意代数形式）→ 真值表（功能本身唯一）；
  2. 与"因式分解模板库"做等价匹配：允许输入排列、输入取反、输出取反；
     模板形状 = 单层 AND/OR/XOR + 两层 AOI/AO/OAI/OA（按整数分拆生成）+ MUX；
  3. 取匹配中总算子数最小者（确定性 tie-break）；
     输入取反每个补 1 个 not（物理上是外接/内置反相器，两边才公平）。
同一布尔功能（含引脚极性差异）⇒ 同一计数；库粒度差（一个 AOI 顶三个分立门）
在算子空间自动抵消。>6 输入的少数单元退回规范 SOP（确定性，记录覆盖率）。

约定（论文写明）：
  - 原语 {not, and, or, xor, mux} 全按二输入计数；buf = 0 算子（恒等）。
  - 时序单元 = 1 seq，不计组合算子（输出 function 只有内部状态 IQ/IQN）。
  - 多输出单元（FA 的 S/CO）各输出 function 分别折算后求和。
  - 模板深度 = 因式分解的层级数（AOI21=3 层：与→或→非），用于库无关逻辑深度。
  - 黑盒桩/未知单元按命名模式给语义后走同一管线。

用法：
  ops_map = load_lib_ops("libs/liberty/nangate45/xxx.lib")
  ops_map.update(generic_cell_ops())
  info = cell_ops(ops_map, "AOI21_X1")  # {'ops': {'and':1,'or':1,'not':1}, 'depth': 3, ...}
"""
import re
from collections import defaultdict
from itertools import permutations

# ---------------------------------------------------------------- 表达式解析

TOKEN_RE = re.compile(r"\s*(?:(?P<id>[A-Za-z_][A-Za-z_0-9[\]]*)|(?P<op>[!&*|^+()]))")


class ExprParser:
    """递归下降：or(+,|) < xor(^) < and(*,&) < not(!) < 原子"""

    def __init__(self, text):
        self.toks = []
        i = 0
        while i < len(text):
            m = TOKEN_RE.match(text, i)
            if not m:
                if text[i:].strip() == "":
                    break
                raise ValueError(f"bad char {text[i:]!r} in {text!r}")
            i = m.end()
            self.toks.append(("id", m.group("id")) if m.group("id") else ("op", m.group("op")))
        self.pos = 0

    def _peek(self):
        return self.toks[self.pos] if self.pos < len(self.toks) else (None, None)

    def _eat(self, val):
        if self._peek()[1] != val:
            raise ValueError(f"expected {val!r} got {self._peek()[1]!r}")
        self.pos += 1

    def parse(self):
        node = self._or()
        if self.pos != len(self.toks):
            raise ValueError(f"trailing tokens at {self.toks[self.pos:]}")
        return node

    def _or(self):
        node = self._xor()
        while self._peek()[1] in ("|", "+"):
            self.pos += 1
            node = ("or", node, self._xor())
        return node

    def _xor(self):
        node = self._and()
        while self._peek()[1] == "^":
            self.pos += 1
            node = ("xor", node, self._and())
        return node

    def _and(self):
        node = self._not()
        while self._peek()[1] in ("&", "*"):
            self.pos += 1
            node = ("and", node, self._not())
        return node

    def _not(self):
        if self._peek()[1] == "!":
            self.pos += 1
            return ("not", self._not())
        return self._atom()

    def _atom(self):
        t, v = self._peek()
        if v == "(":
            self.pos += 1
            node = self._or()
            self._eat(")")
            return node
        if t == "id":
            self.pos += 1
            return ("id", v)
        raise ValueError(f"unexpected token {v!r}")


# 电源/地 token：按常量求值（VPWR=1, VGND=0）；内部状态 token：时序输出，跳过
CONST_ENV = {"VPWR": 1, "VGND": 0, "VDD": 1, "VSS": 0, "VNB": 0, "VPB": 1,
             "KAPWR": 1, "LOWLVPWR": 1}
STATE_TOKENS = {"IQ", "IQN", "IQ_N"}
POWER_TOKENS = set(CONST_ENV) | STATE_TOKENS

MAX_CATALOG_VARS = 6   # 模板匹配上限；超过退回规范 SOP


def _eval_ast(node, env):
    kind = node[0]
    if kind == "id":
        if node[1] in env:
            return env[node[1]]
        return CONST_ENV[node[1]]
    if kind == "not":
        return 1 - _eval_ast(node[1], env)
    a, b = _eval_ast(node[1], env), _eval_ast(node[2], env)
    if kind == "and":
        return a & b
    if kind == "or":
        return a | b
    return a ^ b


def truth_table(ast, vars_):
    """所有赋值下的函数值；赋值索引的第 i 位 = vars_[i] 的值。"""
    tt = []
    for idx in range(1 << len(vars_)):
        env = {v: (idx >> i) & 1 for i, v in enumerate(vars_)}
        tt.append(_eval_ast(ast, env))
    return tt


# ---------------------------------------------------------------- 模板库

def _int_partitions(n, max_part=None):
    """n 的整数分拆（降序），如 3 → [[3],[2,1],[1,1,1]]。"""
    if max_part is None:
        max_part = n
    if n == 0:
        yield []
        return
    for first in range(min(n, max_part), 0, -1):
        for rest in _int_partitions(n - first, first):
            yield [first] + rest


def _shape_eval(groups, mode, env_vals):
    """两层形状求值。mode: 'AO'=或(与组), 'AOI'=取反, 'OA'=与(或组), 'OAI'。"""
    if mode in ("AO", "AOI"):
        acc = 0
        for g in groups:
            t = 1
            for v in g:
                t &= env_vals[v]
            acc |= t
    else:
        acc = 1
        for g in groups:
            t = 0
            for v in g:
                t |= env_vals[v]
            acc &= t
    return 1 - acc if mode in ("AOI", "OAI") else acc


def _ops_cost(ops):
    return (sum(ops.values()), sorted(ops.items()))


def _build_templates(k):
    """k 输入的因式分解模板 → {真值表: (ops, depth)}。"""
    tpls = {}
    vars_ = list(range(k))

    def add(tt, ops, depth):
        key = tuple(tt)
        clean = {k: v for k, v in ops.items() if v > 0}
        cand = (clean, depth)
        if key not in tpls or _ops_cost(cand[0]) < _ops_cost(tpls[key][0]):
            tpls[key] = cand

    for parts in _int_partitions(k):
        groups, i = [], 0
        for sz in parts:
            groups.append(vars_[i:i + sz])
            i += sz
        for mode in ("AO", "AOI", "OA", "OAI"):
            ops = defaultdict(int)
            if mode in ("AO", "AOI"):
                ops["and"] = sum(max(0, len(g) - 1) for g in groups)
                ops["or"] = max(0, len(groups) - 1)
                n_lev = (1 if any(len(g) >= 2 for g in groups) else 0) \
                    + (1 if len(groups) >= 2 else 0)
            else:
                ops["or"] = sum(max(0, len(g) - 1) for g in groups)
                ops["and"] = max(0, len(groups) - 1)
                n_lev = (1 if any(len(g) >= 2 for g in groups) else 0) \
                    + (1 if len(groups) >= 2 else 0)
            if mode in ("AOI", "OAI"):
                ops["not"] += 1
                n_lev += 1
            tt = []
            for idx in range(1 << k):
                env = {v: (idx >> v) & 1 for v in vars_}
                tt.append(_shape_eval(groups, mode, env))
            add(tt, ops, n_lev)
    # XOR / XNOR
    if k >= 2:
        tt = [bin(i).count("1") & 1 for i in range(1 << k)]
        add(tt, {"xor": k - 1}, 1)
        add([1 - v for v in tt], {"xor": k - 1, "not": 1}, 2)
    # MUX（k==3）：f = s ? a : b（含输出取反）
    if k == 3:
        for s in range(3):
            a, b = [v for v in range(3) if v != s]
            tt = []
            for idx in range(1 << k):
                sel = (idx >> s) & 1
                tt.append((idx >> (b if sel else a)) & 1)
            add(tt, {"mux": 1}, 2)
            add([1 - v for v in tt], {"mux": 1, "not": 1}, 3)
    return tpls


_TEMPLATES = {k: _build_templates(k) for k in range(1, MAX_CATALOG_VARS + 1)}


def _match_catalog(tt, k):
    """真值表 ↔ 模板库等价匹配（输入排列/取反 + 模板自带输出极性）。
    返回 (ops, depth)；取总算子数最小的匹配，输入取反每个补 1 not。"""
    tpls = _TEMPLATES[k]
    best = None
    for perm in permutations(range(k)):
        for mask in range(1 << k):
            tt2 = [0] * (1 << k)
            for y in range(1 << k):
                x = 0
                for j in range(k):
                    i = perm[j]
                    bit = (y >> j) & 1
                    if (mask >> i) & 1:
                        bit ^= 1
                    x |= bit << i
                tt2[y] = tt[x]
            hit = tpls.get(tuple(tt2))
            if hit is None:
                continue
            ops = dict(hit[0])
            n_compl = bin(mask).count("1")
            if n_compl:
                ops["not"] = ops.get("not", 0) + n_compl
            depth = hit[1] + (1 if n_compl else 0)
            cand = (ops, depth)
            key = (_ops_cost(ops), depth)
            if best is None or key < best[0]:
                best = (key, cand)
    return best[1] if best else None


# ---------------------------------------------------------------- 规范 SOP（降级）

def _prime_implicants(minterms, k):
    terms = {(m, 0) for m in minterms}
    pis = set()
    while terms:
        used = set()
        nxt = set()
        tl = sorted(terms)
        for i in range(len(tl)):
            b1, m1 = tl[i]
            for j in range(i + 1, len(tl)):
                b2, m2 = tl[j]
                if m1 != m2:
                    continue
                diff = (b1 ^ b2) & ~m1
                if diff.bit_count() == 1:
                    used.add((b1, m1))
                    used.add((b2, m2))
                    nxt.add(((b1 & ~diff) & ~(m1 | diff), m1 | diff))
        for t in terms:
            if t not in used:
                pis.add(t)
        terms = nxt
    return sorted(pis)


def _canonical_sop(tt, k):
    """确定性规范 SOP 覆盖 → (ops, depth)。仅用于 >6 输入的降级路径。"""
    minterms = [i for i in range(1 << k) if tt[i]]
    if not minterms or len(minterms) == (1 << k):
        return {}, 0
    pis = _prime_implicants(minterms, k)
    cover = {m: [p for p in pis if ((m ^ p[0]) & ~p[1]) == 0] for m in minterms}
    chosen = []
    remaining = set(minterms)
    for m in list(remaining):
        opts = cover[m]
        if len(opts) == 1 and opts[0] not in chosen:
            chosen.append(opts[0])
            remaining -= {mm for mm in remaining if ((mm ^ opts[0][0]) & ~opts[0][1]) == 0}
    while remaining:
        best, best_key = None, None
        for p in pis:
            if p in chosen:
                continue
            n = sum(1 for m in remaining if ((m ^ p[0]) & ~p[1]) == 0)
            if n == 0:
                continue
            key = (-n, p[1], p[0])
            if best_key is None or key < best_key:
                best_key, best = key, p
        if best is None:
            break
        chosen.append(best)
        remaining -= {m for m in remaining if ((m ^ best[0]) & ~best[1]) == 0}
    ops = defaultdict(int)
    max_term_depth = 0
    n_neg = 0
    for bits, dmask in chosen:
        lits = []
        neg = 0
        for i in range(k):
            if (dmask >> i) & 1:
                continue
            if (bits >> i) & 1:
                lits.append(1)
            else:
                lits.append(0)
                neg += 1
        n_neg += neg
        if len(lits) >= 2:
            ops["and"] += len(lits) - 1
        max_term_depth = max(max_term_depth, (1 if neg else 0) + max(0, len(lits) - 1))
    if len(chosen) >= 2:
        ops["or"] += len(chosen) - 1
    if n_neg:
        ops["not"] += n_neg
    return dict(ops), max_term_depth + (1 if len(chosen) >= 2 else 0)


# ---------------------------------------------------------------- 入口

_PROFILE_CACHE = {}


def cell_op_profile(expr):
    """function 表达式 → (ops, depth)。真值表模板匹配，与代数写法无关。"""
    if expr in _PROFILE_CACHE:
        return _PROFILE_CACHE[expr]
    ast = ExprParser(expr).parse()

    def collect(node, acc):
        if node[0] == "id":
            acc.add(node[1])
        elif node[0] == "not":
            collect(node[1], acc)
        else:
            collect(node[1], acc)
            collect(node[2], acc)

    names = set()
    collect(ast, names)
    vars_ = sorted(names - POWER_TOKENS)
    if not vars_:
        return {}, 0
    k = len(vars_)
    tt = truth_table(ast, vars_)
    if k <= MAX_CATALOG_VARS:
        res = _match_catalog(tt, k) or _canonical_sop(tt, k)
    else:
        res = _canonical_sop(tt, k)
    _PROFILE_CACHE[expr] = res
    return res


# ---------------------------------------------------------------- Liberty 解析

CELL_RE = re.compile(r"\bcell\s*\(\s*\"?([A-Za-z_][A-Za-z_0-9.\[\]]*)\"?\s*\)\s*\{")
PIN_RE = re.compile(r"\bpin\s*\(\s*\"?([A-Za-z_][A-Za-z_0-9[\]]*)\"?\s*\)\s*\{")
DIR_RE = re.compile(r"\bdirection\s*:\s*\"?(\w+)\"?\s*;")
FUNC_RE = re.compile(r"\bfunction\s*:\s*\"([^\"]*)\"\s*;")
FF_RE = re.compile(r"\b(?:ff|latch)\s*\(")


def _block_end(text, start):
    depth = 0
    i = start
    n = len(text)
    while i < n:
        c = text[i]
        if c == '"':
            i += 1
            while i < n and text[i] != '"':
                i += 2 if text[i] == "\\" else 1
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError("unbalanced braces")


def load_lib_ops(lib_path, stats=None):
    """解析 .lib → {cell_name: {'ops': {...}, 'depth': int, 'seq': bool}}。"""
    text = open(lib_path, encoding="utf-8", errors="replace").read()
    out = {}
    n_out, n_ok = 0, 0
    for m in CELL_RE.finditer(text):
        name = m.group(1)
        brace = text.index("{", m.end() - 1)
        end = _block_end(text, brace)
        body = text[m.end():end]
        info = {"ops": defaultdict(int), "depth": 0,
                "seq": bool(FF_RE.search(body))}
        for pm in PIN_RE.finditer(body):
            pbrace = body.index("{", pm.end() - 1)
            pend = _block_end(body, pbrace)
            pbody = body[pm.end():pend]
            dm = DIR_RE.search(pbody)
            if not dm or dm.group(1) != "output":
                continue
            n_out += 1
            fm = FUNC_RE.search(pbody)
            if not fm:
                continue
            expr = fm.group(1)
            ids = set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", expr))
            if ids and (ids & STATE_TOKENS or ids <= set(CONST_ENV)):
                continue  # 纯内部状态/电源 function → 无组合算子
            try:
                ops, depth = cell_op_profile(expr)
            except ValueError:
                continue
            n_ok += 1
            for kk, v in ops.items():
                info["ops"][kk] += v
            info["depth"] = max(info["depth"], depth)
        out[name] = info
    if stats is not None:
        stats["out_pins"] = stats.get("out_pins", 0) + n_out
        stats["ok"] = stats.get("ok", 0) + n_ok
    return out


# ---------------------------------------------------------------- yosys 原语

_GENERIC_FUNCS = {
    "NOT": "!A", "BUF": "A", "TBUF": "A",
    "AND": "A&B", "NAND": "!(A&B)", "OR": "A|B", "NOR": "!(A|B)",
    "XOR": "A^B", "XNOR": "!(A^B)",
    "ANDNOT": "A&!B", "ORNOT": "A|!B",
    "MUX": "(S&B)|(!S&A)", "NMUX": "!((S&B)|(!S&A))",
    "AOI3": "!((A&B)|C)", "OAI3": "!((A|B)&C)",
    "AOI4": "!((A&B)|(C&D))", "OAI4": "!((A|B)&(C|D))",
}
_GENERIC_SEQ = ("DFF_P", "DFF_N", "DFFE_P", "DFFE_N", "DFFSR_P", "DFFSR_N",
                "SDFF_P", "SDFF_N", "DLATCH_P", "DLATCH_N")

_GENERIC_CACHE = {}


def generic_cell_ops():
    if not _GENERIC_CACHE:
        for name, expr in _GENERIC_FUNCS.items():
            try:
                ops, depth = cell_op_profile(expr)
            except ValueError:
                ops, depth = {}, 0
            _GENERIC_CACHE[f"$_{name}_"] = {"ops": ops, "depth": depth, "seq": False}
        for dff in _GENERIC_SEQ:
            _GENERIC_CACHE[f"$_{dff}_"] = {"ops": {}, "depth": 0, "seq": True}
    return _GENERIC_CACHE


# TRIT 门级网表自造单元名（黑盒桩）按命名模式给语义，再走同一管线：
# nnd2s1=NAND2 / nor5s1=NOR5 / i1s1=INV / hi1s1=常数高 / xor2s1=XOR2，s1=驱动档
_STUB_RE = re.compile(r"^(?P<base>[a-zA-Z]+?)(?P<num>\d+)")
_STUB_BASE = {
    "nnd": "nand", "nand": "nand", "and": "and", "nd": "nand",
    "nor": "nor", "or": "or", "xor": "xor", "xnor": "xnor",
    "inv": "inv", "i": "inv", "not": "inv",
    "buf": "buf", "b": "buf",
    "mux": "mux", "mx": "mux",
    "hi": "const", "lo": "const", "tih": "const", "til": "const",
}


def stub_cell_ops(cell_type):
    m = _STUB_RE.match(cell_type)
    if not m:
        return None
    kind = _STUB_BASE.get(m.group("base").lower())
    if kind is None:
        return None
    if kind == "const":
        return {"ops": {}, "depth": 0, "seq": False}
    if kind == "inv":
        expr = "!A"
    elif kind == "buf":
        expr = "A"
    elif kind == "mux":
        expr = "(S&B)|(!S&A)"
    else:
        op, inv = {"and": ("&", 0), "nand": ("&", 1), "or": ("|", 0),
                   "nor": ("|", 1), "xor": ("^", 0), "xnor": ("^", 1)}[kind]
        k = int(m.group("num")) if int(m.group("num")) >= 2 else 2
        inputs = "ABCDEF"[:max(2, min(6, k))]
        core = op.join(inputs)
        expr = f"!({core})" if inv else core
    try:
        ops, depth = cell_op_profile(expr)
    except ValueError:
        return None
    return {"ops": ops, "depth": depth, "seq": False}


def cell_ops(ops_map, cell_type):
    """查表 + 兜底：返回 {'ops': {..}, 'depth': d, 'seq': bool}；未知返回 None。"""
    if cell_type in ops_map:
        return ops_map[cell_type]
    if cell_type.startswith("$_"):
        return generic_cell_ops().get(cell_type)
    if cell_type.startswith("$"):
        return {"ops": {}, "depth": 0, "seq": False}  # $scopeinfo 等元数据
    return stub_cell_ops(cell_type)
