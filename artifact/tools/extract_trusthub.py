#!/usr/bin/env python3
"""extract_trusthub.py —— 解 Trust-Hub 镜像仓库里的基准压缩包。
zip 用标准库解；rar 报告出来交给外部工具。解到各变体目录下的 extracted/ 子目录，原包保留。"""
import os
import re
import sys
import subprocess
import zipfile

ROOT = r"./data\Trust-Hub\chip-level-trojan"
UNRAR = os.environ.get("UNRAR", r"C:\Program Files\WinRAR\UnRAR.exe")  # override via env var

ok, fail, rar_ok, rar_fail = 0, 0, 0, 0


def unrar_extract(src, dest):
    os.makedirs(dest, exist_ok=True)
    r = subprocess.run([UNRAR, "x", "-o+", src, dest + "\\"],
                       capture_output=True, text=True, timeout=600)
    return r.returncode == 0


for dirpath, _dirnames, filenames in os.walk(ROOT):
    for fn in sorted(filenames):
        low = fn.lower()
        src = os.path.join(dirpath, fn)
        dest = os.path.join(dirpath, "extracted")
        if low.endswith(".zip"):
            try:
                with zipfile.ZipFile(src) as z:
                    z.extractall(dest)
                ok += 1
            except Exception:
                # 假 zip（内容其实是 rar）→ UnRAR 按内容识别
                if os.path.isfile(UNRAR) and unrar_extract(src, dest):
                    ok += 1
                    print("假 zip 已由 UnRAR 解开:", src.replace(ROOT, ""))
                else:
                    fail += 1
                    print("FAIL:", src.replace(ROOT, ""))
        elif low.endswith(".rar"):
            # 分卷只解 part01（UnRAR 自动认其余卷）；独立包逐个解
            if re.search(r"\.part0*1\.rar$", low):
                if unrar_extract(src, dest):
                    rar_ok += 1
                else:
                    rar_fail += 1
                    print("FAIL rar:", src.replace(ROOT, ""))
            elif re.search(r"\.part\d+\.rar$", low):
                continue  # 非首卷，跳过
            else:
                if unrar_extract(src, dest):
                    rar_ok += 1
                else:
                    rar_fail += 1
                    print("FAIL rar:", src.replace(ROOT, ""))

print(f"zip 提取成功: {ok}  失败: {fail}")
print(f"rar 提取成功: {rar_ok}  失败: {rar_fail}")

# 报告解出的 Verilog 情况
v_count = 0
families = {}
for dirpath, _d, filenames in os.walk(ROOT):
    for fn in filenames:
        if fn.lower().endswith((".v", ".vhd", ".vhdl")):
            v_count += 1
            fam = os.path.relpath(dirpath, ROOT).split(os.sep)[0]
            families[fam] = families.get(fam, 0) + 1
print(f"\n解出 HDL 文件: {v_count}")
for fam, n in sorted(families.items()):
    print(f"  {fam}: {n}")
