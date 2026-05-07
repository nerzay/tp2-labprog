"""
pipeline.py — Pipeline de pré-processamento de texto para SLMs
Etapa 1 do projeto LabProg — Extração, limpeza e segmentação de documentos.
"""

import re
import os
from collections import Counter

import fitz                     # PyMuPDF : extração de PDF
from docx import Document       # python-docx : extração de DOCX
from langdetect import detect, DetectorFactory

# Semente fixa para tornar a deteção de língua reprodutível
DetectorFactory.seed = 42


# ─────────────────────────────────────────────
# EXTRAÇÃO DO TEXTO EM BRUTO
# ─────────────────────────────────────────────

def extract_text(filepath: str) -> str:
    """Encaminha para a função correta consoante a extensão do ficheiro."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(filepath)
    elif ext == ".docx":
        return extract_text_from_docx(filepath)
    elif ext == ".txt":
        return extract_text_from_txt(filepath)
    else:
        raise ValueError(f"Formato não suportado: {ext}")


def extract_text_from_pdf(filepath: str) -> str:
    """
    Extrai o texto de um PDF página a página com o PyMuPDF.
    Insere marcadores [PAGE n] para facilitar a deteção
    de cabeçalhos e rodapés repetidos.
    """
    doc = fitz.open(filepath)
    pages = []
    for i, page in enumerate(doc):
        content = page.get_text("text")   # "text" preserva a ordem de leitura
        pages.append(f"[PAGE {i + 1}]\n{content}")
    doc.close()
    return "\n\n".join(pages)


def extract_text_from_docx(filepath: str) -> str:
    """
    Extrai o texto de um ficheiro DOCX parágrafo a parágrafo.
    Preserva as quebras de parágrafo (dupla mudança de linha).
    """
    doc = Document(filepath)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def extract_text_from_txt(filepath: str) -> str:
    """
    Lê um ficheiro TXT experimentando vários encodings comuns.
    Usa utf-8 com substituição de caracteres desconhecidos como último recurso.
    """
    for encoding in ("utf-8", "latin-1", "cp1252", "utf-16"):
        try:
            with open(filepath, "r", encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, LookupError):
            continue
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


# ─────────────────────────────────────────────
# ETAPAS DO PIPELINE DE LIMPEZA
# ─────────────────────────────────────────────

def remove_artifacts(text: str) -> str:
    """
    Remove artefactos comuns em textos extraídos de PDF:
    - Caracteres de controlo (exceto \\n, \\t)
    - Sequências repetidas de símbolos (linhas separadoras)
    - Linhas compostas apenas por caracteres não alfanuméricos
    """
    # Caracteres de controlo indesejados
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Sequências longas de símbolos idênticos (ex: ---------, ........)
    text = re.sub(r"([^\w\s])\1{4,}", r"\1\1\1", text)
    # Linhas contendo apenas caracteres não alfanuméricos
    text = re.sub(r"(?m)^[^\w\n]{5,}$", "", text)
    return text


def detect_remove_headers_footers(text: str) -> str:
    """
    Deteta linhas repetidas em várias páginas (cabeçalhos / rodapés)
    e remove-as. Limiar: linha que aparece em ≥ 40 % das páginas.
    """
    lines = text.split("\n")

    # Contar as ocorrências de cada linha não vazia
    counter: Counter = Counter()
    for line in lines:
        stripped = line.strip()
        if stripped and len(stripped) > 3:   # ignorar linhas muito curtas
            counter[stripped] += 1

    # Estimar o número de páginas com os marcadores [PAGE n]
    page_count = max(text.count("[PAGE "), 1)
    threshold = max(2, int(page_count * 0.4))

    repeated = {line for line, count in counter.items() if count >= threshold}

    filtered = [l for l in lines if l.strip() not in repeated]
    return "\n".join(filtered)


def fix_line_breaks(text: str) -> str:
    """
    Remove quebras de linha incorretas no interior das frases.
    Uma linha é fundida com a seguinte se:
      - não terminar com sinal de pontuação final ( . ! ? : ; )
      - a linha seguinte começar com minúscula
    """
    lines = text.split("\n")
    result = []
    i = 0
    while i < len(lines):
        current = lines[i].rstrip()
        if (
            i + 1 < len(lines)
            and current
            and current[-1] not in ".!?:;"
            and lines[i + 1].strip()
            and lines[i + 1].lstrip()[0:1].islower()
        ):
            # Fundir a linha atual com a seguinte
            lines[i + 1] = current + " " + lines[i + 1].lstrip()
        else:
            result.append(current)
        i += 1
    return "\n".join(result)


def reconstruct_paragraphs(text: str) -> str:
    """
    Agrupa linhas consecutivas em parágrafos coerentes.
    Os blocos separados por linhas vazias são tratados de forma independente.
    Os marcadores [PAGE n] também iniciam uma nova secção.
    """
    # Separar os blocos por linhas vazias ou marcadores de página
    blocks = re.split(r"\n{2,}|\[PAGE \d+\]", text)
    reconstructed = []
    for block in blocks:
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        if lines:
            # Juntar as linhas do bloco num único parágrafo
            reconstructed.append(" ".join(lines))
    return "\n\n".join(reconstructed)


def normalize_spaces(text: str) -> str:
    """Substitui tabulações e espaços múltiplos por um único espaço."""
    text = text.replace("\t", " ")
    text = re.sub(r" {2,}", " ", text)
    # Remover espaços no início/fim de cada linha
    lines = [l.strip() for l in text.split("\n")]
    return "\n".join(lines)


def normalize_punctuation(text: str) -> str:
    """
    Corrige problemas de pontuação comuns:
    - Espaço antes dos sinais de pontuação
    - Ausência de espaço após pontuação
    - Aspas tipográficas → aspas direitas
    - Pontos múltiplos → reticências
    """
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"([.,;:!?])([A-ZÀ-Ü])", r"\1 \2", text)
    # Aspas francesas e curvas
    for old, new in [("«", '"'), ("»", '"'), ("‘", "'"), ("’", "'"),
                     ("“", '"'), ("”", '"')]:
        text = text.replace(old, new)
    text = re.sub(r"\.{4,}", "...", text)
    return text


def normalize_special_chars(text: str) -> str:
    """
    Substitui caracteres Unicode especiais pelos seus equivalentes ASCII:
    travessões, espaços não separáveis, largura zero, BOM, etc.
    """
    substitutions = {
        "–": "-",    # travessão médio
        "—": "-",    # travessão longo
        "…": "...",  # reticências
        " ": " ",    # espaço não separável
        "​": "",     # espaço de largura zero
        "﻿": "",     # BOM (byte order mark)
        "•": "-",    # marcador redondo
        "’": "'",    # apóstrofo curvo
    }
    for char, replacement in substitutions.items():
        text = text.replace(char, replacement)
    return text


# ─────────────────────────────────────────────
# ORQUESTRADOR DO PIPELINE
# ─────────────────────────────────────────────

def apply_pipeline(text: str, options: dict) -> tuple[str, list]:
    """
    Aplica as etapas de limpeza selecionadas em `options`.
    Devolve (texto_limpo, registo_das_etapas).
    Cada entrada do registo indica o nome da etapa e os
    caracteres removidos durante essa etapa.
    """
    log = []
    current = text

    steps = [
        ("remove_artifacts",        "Remoção de artefactos",                  remove_artifacts),
        ("remove_headers_footers",  "Remoção de cabeçalhos / rodapés",        detect_remove_headers_footers),
        ("fix_line_breaks",         "Correção de quebras de linha",           fix_line_breaks),
        ("reconstruct_paragraphs",  "Reconstrução de parágrafos",             reconstruct_paragraphs),
        ("normalize_spaces",        "Normalização de espaços",                normalize_spaces),
        ("normalize_punctuation",   "Normalização da pontuação",              normalize_punctuation),
        ("normalize_special_chars", "Normalização de caracteres especiais",   normalize_special_chars),
    ]

    for key, label, func in steps:
        if options.get(key, True):
            before = current
            current = func(current)
            log.append({
                "step": label,
                "chars_removed": len(before) - len(current),
            })

    return current, log


# ─────────────────────────────────────────────
# SEGMENTAÇÃO EM CHUNKS
# ─────────────────────────────────────────────

def segment_chunks(text: str, chunk_size: int = 512) -> list[str]:
    """
    Divide o texto em segmentos de aproximadamente `chunk_size` caracteres.
    A divisão é feita de preferência nas fronteiras de frases (. ! ?)
    para preservar a coerência semântica de cada chunk.
    """
    if not text.strip():
        return []

    chunks: list[str] = []
    # Dividir primeiro nas fronteiras de frases
    sentences = re.split(r"(?<=[.!?])\s+", text)

    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk += (" " if current_chunk else "") + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # Se a frase sozinha ultrapassar o tamanho, dividir por palavras
            if len(sentence) > chunk_size:
                words = sentence.split()
                temp = ""
                for word in words:
                    if len(temp) + len(word) + 1 <= chunk_size:
                        temp += (" " if temp else "") + word
                    else:
                        if temp:
                            chunks.append(temp.strip())
                        temp = word
                current_chunk = temp
            else:
                current_chunk = sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# ─────────────────────────────────────────────
# DETEÇÃO DE LÍNGUA
# ─────────────────────────────────────────────

LANG_NAMES = {
    "fr": "Français", "en": "English", "es": "Español",
    "de": "Deutsch",  "it": "Italiano", "pt": "Português",
    "nl": "Nederlands", "ru": "Русский", "ar": "العربية",
    "zh-cn": "Chinês (simplificado)", "ja": "Japonês",
}


def detect_language(text: str) -> dict:
    """
    Deteta a língua do texto com o langdetect.
    Utiliza os primeiros 2000 caracteres para acelerar a deteção.
    Devolve um dict {"code": "pt", "name": "Português"}.
    """
    try:
        sample = text[:2000].strip()
        if not sample:
            return {"code": "unknown", "name": "Desconhecida"}
        code = detect(sample)
        return {"code": code, "name": LANG_NAMES.get(code, code.upper())}
    except Exception:
        return {"code": "unknown", "name": "Desconhecida"}


# ─────────────────────────────────────────────
# GERAÇÃO DE PROMPTS & PAYLOAD SLM
# ─────────────────────────────────────────────

PROMPT_TEMPLATES = {
    "pt": {
        "general":   "Analisa e resume o seguinte texto em português:\n\n{text}",
        "summarize": "Fornece um resumo conciso (5 a 7 frases) do seguinte texto:\n\n{text}",
        "extract":   "Extrai as informações-chave (entidades, datas, factos principais) do seguinte texto:\n\n{text}",
        "qa":        "Com base no seguinte texto, formula e responde a 3 perguntas importantes:\n\n{text}",
    },
    "fr": {
        "general":   "Analyse et résume le texte suivant en français :\n\n{text}",
        "summarize": "Fournis un résumé concis (5 à 7 phrases) du texte suivant :\n\n{text}",
        "extract":   "Extrais les informations clés (entités, dates, faits principaux) du texte suivant :\n\n{text}",
        "qa":        "En te basant sur le texte suivant, formule et réponds à 3 questions importantes :\n\n{text}",
    },
    "en": {
        "general":   "Analyze and summarize the following text in English:\n\n{text}",
        "summarize": "Provide a concise summary (5 to 7 sentences) of the following text:\n\n{text}",
        "extract":   "Extract the key information (entities, dates, main facts) from the following text:\n\n{text}",
        "qa":        "Based on the following text, formulate and answer 3 important questions:\n\n{text}",
    },
}

# Recorre ao português para línguas sem template dedicado
_FALLBACK_LANG = "pt"


def generate_prompt(text: str, language, norm_type: str = "general") -> str:
    """
    Gera um prompt adaptado à língua detetada e ao tipo de normalização.
    `language` pode ser um código string ("pt") ou um dict {"code": "pt", ...}.
    """
    lang_code = language if isinstance(language, str) else language.get("code", _FALLBACK_LANG)
    templates = PROMPT_TEMPLATES.get(lang_code, PROMPT_TEMPLATES[_FALLBACK_LANG])
    template = templates.get(norm_type, templates["general"])
    return template.format(text=text)


def build_slm_payload(prompt: str) -> dict:
    """
    Constrói o payload JSON para a API SLM.
    Formato esperado por https://reality.utad.net/slm
    """
    return {
        "model": "llama-3.2-1b-instruct",
        "messages": [
            {"role": "user", "content": prompt}
        ],
    }
