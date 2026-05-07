"""
app.py — Servidor Flask para o pipeline de pré-processamento de texto SLM
Etapa 1 do projeto LabProg

Arranque : python app.py
Interface  : http://localhost:5000
"""

import os
import json
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from pipeline import (
    extract_text,
    apply_pipeline,
    segment_chunks,
    detect_language,
    generate_prompt,
    build_slm_payload,
)

# ─────────────────────────────────────────────
# Configuração da aplicação
# ─────────────────────────────────────────────

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024   # 32 Mo maximum

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}


def allowed_file(filename: str) -> bool:
    """Verifica se o ficheiro tem uma extensão permitida."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ─────────────────────────────────────────────
# Rotas
# ─────────────────────────────────────────────

@app.route("/")
def index():
    """Página principal da interface."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    """
    Recebe um ficheiro (PDF / DOCX / TXT), guarda-o em /uploads
    e extrai e devolve o texto em bruto.
    """
    if "file" not in request.files:
        return jsonify({"error": "Nenhum ficheiro fornecido."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Nenhum ficheiro selecionado."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": "Formato não suportado. Utilize um ficheiro PDF, DOCX ou TXT."
        }), 400

    # Garantir que o nome do ficheiro é seguro antes de o guardar
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        raw_text = extract_text(filepath)
    except Exception as e:
        return jsonify({"error": f"Erro durante a extração: {str(e)}"}), 500

    # Estatísticas básicas sobre o texto em bruto
    stats = {
        "chars": len(raw_text),
        "words": len(raw_text.split()),
        "lines": raw_text.count("\n") + 1,
    }

    return jsonify({
        "success": True,
        "filename": filename,
        "raw_text": raw_text,
        "stats": stats,
    })


@app.route("/process", methods=["POST"])
def process_text():
    """
    Aplica o pipeline de limpeza ao texto recebido.
    As opções ativas são passadas no corpo JSON.
    Devolve o texto limpo, o registo das etapas,
    os chunks e a língua detetada.
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Corpo JSON inválido."}), 400

    raw_text   = data.get("raw_text", "")
    options    = data.get("options", {})
    chunk_size = int(data.get("chunk_size", 512))

    if not raw_text.strip():
        return jsonify({"error": "O texto fornecido está vazio."}), 400

    try:
        cleaned_text, steps_log = apply_pipeline(raw_text, options)
        chunks   = segment_chunks(cleaned_text, chunk_size)
        language = detect_language(cleaned_text)
    except Exception as e:
        return jsonify({"error": f"Erro durante o processamento: {str(e)}"}), 500

    stats_cleaned = {
        "chars": len(cleaned_text),
        "words": len(cleaned_text.split()),
        "lines": cleaned_text.count("\n") + 1,
    }

    return jsonify({
        "success": True,
        "cleaned_text": cleaned_text,
        "steps_log":    steps_log,
        "chunks":       chunks,
        "chunk_count":  len(chunks),
        "language":     language,
        "stats":        stats_cleaned,
    })


@app.route("/generate-payload", methods=["POST"])
def generate_payload():
    """
    Gera o prompt e o payload JSON pronto a enviar para a API SLM
    para o chunk selecionado.
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Corpo JSON inválido."}), 400

    chunks      = data.get("chunks", [])
    chunk_index = int(data.get("chunk_index", 0))
    language    = data.get("language", {"code": "pt", "name": "Português"})
    norm_type   = data.get("norm_type", "general")

    if not chunks:
        return jsonify({"error": "Nenhum chunk disponível."}), 400

    if chunk_index >= len(chunks):
        chunk_index = 0

    content = chunks[chunk_index]

    try:
        prompt  = generate_prompt(content, language, norm_type)
        payload = build_slm_payload(prompt)
    except Exception as e:
        return jsonify({"error": f"Erro durante a geração: {str(e)}"}), 500

    return jsonify({
        "success": True,
        "chunk_index": chunk_index,
        "prompt":      prompt,
        "payload":     payload,
    })


# ─────────────────────────────────────────────
# Ponto de entrada
# ─────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    print("=== Pré-processador SLM — LabProg Etapa 1 ===")
    print("Interface disponível em: http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
