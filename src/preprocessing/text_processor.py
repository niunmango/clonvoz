"""
Módulo de procesamiento de texto y adaptación fonética rioplatense para VoxCPM2.
"""

import re
from typing import List


def convert_to_rioplatense(text: str, aspirate_s: bool = False) -> str:
    """
    Transforma texto en español estándar a fonética rioplatense (sheísmo y opcional aspiración de /s/).

    Reglas fonéticas:
    1. 'll' -> 'sh' (ej: 'lluvia' -> 'shuvia', 'calle' -> 'cashe')
    2. 'y' antes de vocal -> 'sh' (ej: 'yo' -> 'sho', 'playa' -> 'plasha', 'ayer' -> 'asher')
       Evita modificar palabras prestadas como 'youtube' o la conjunción 'y' aislada.
    3. (Opcional) Aspiración de /s/ ante consonante: 'está' -> 'ehtá', 'mismo' -> 'mihmo'.
    """
    if not text:
        return ""

    # Proteger extranjerismos y términos reservados (ej: YouTube, Byte)
    protected = {}
    tokens = ["YouTube", "youtube", "Youtube", "Byte", "byte", "Bytes", "bytes", "Python", "python"]
    for i, tok in enumerate(tokens):
        placeholder = f"__PROT_TOKEN_{i}__"
        if tok in text:
            text = text.replace(tok, placeholder)
            protected[placeholder] = tok

    # 1. Sheísmo: 'll' -> 'sh'
    result = re.sub(r'll', 'sh', text, flags=re.IGNORECASE)

    # 2. 'y' vocálica/intervocálica/inicial antes de vocal -> 'sh'
    result = re.sub(r'(?<![tTyouU])(?<![yY])yl', 'sh', result, flags=re.IGNORECASE)
    result = re.sub(r'(?<![tTyouU])(?<![yY])y(?=[aeiouáéíóú])', 'sh', result, flags=re.IGNORECASE)
    result = re.sub(r'(?<![tT])oo(?=[aeiouáéíóú])', 'o', result, flags=re.IGNORECASE)

    # 3. Aspiración de /s/ ante consonante (si está habilitada)
    if aspirate_s:
        # s seguida de consonante (b, c, d, f, g, j, k, l, m, n, p, q, r, s, t, v, w, x, z)
        consonantes = r'[bcdfghjklmnpqrstvwxzBCDFGHJKLMNPQRSTVWXZ]'
        result = re.sub(rf's(?={consonantes})', 'h', result)
        result = re.sub(rf'S(?={consonantes})', 'H', result)

    # Restaurar términos protegidos
    for placeholder, original in protected.items():
        result = result.replace(placeholder, original)

    return result


def segment_script(script_text: str) -> List[str]:
    """
    Segmenta un guion en bloques de texto separados por líneas en blanco o saltos de párrafo.
    Limpia espacios en blanco y descarta bloques vacíos.
    """
    if not script_text:
        return []

    bloques = [p.strip() for p in re.split(r'\n\s*\n', script_text) if p.strip()]
    return bloques
