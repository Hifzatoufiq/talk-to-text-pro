import os
import tempfile
from datetime import datetime, timezone
from functools import wraps

from flask import (
    Flask, request, render_template, redirect, url_for,
    flash, jsonify, send_file, session
)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

from pymongo import MongoClient
from bson import ObjectId

# TalkToTextPro core
from utils.audio_processing import extract_audio_from_file, transcribe_audio, summarize_text

# OpenAI (for Translator)
from openai import OpenAI

# OCR / PDF / DOCX utils (GeoSpeak-like features)
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from docx import Document
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# OPTIONAL (Windows): set Tesseract installed path; comment out on mac/linux
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -------------------------------------------------------
# Absolute paths (templates/static inside /backend)
# -------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_FOLDER = os.path.join(STATIC_DIR, "uploads")
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a', 'mp4', 'opus', 'ogg', 'webm', 'mov', 'mkv', 'avi', 'aac'}

app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SECRET_KEY"] = "devsecret"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# -------------------------------------------------------
# MongoDB
# -------------------------------------------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = MongoClient(MONGO_URI)
db = client["talktodb"]

users_collection         = db["users"]
notes_collection         = db["notes"]
translations_collection  = db["translations"]
conversions_collection   = db["conversions"]

# -------------------------------------------------------
# OpenAI (HARDCODED KEY, per your request)
# -------------------------------------------------------
OPENAI_API_KEY = ""
oai = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------------------------------------
# Helpers
# -------------------------------------------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def login_required(view_fn):
    @wraps(view_fn)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            flash("Please log in to use Meeting Notes.", "warning")
            return redirect(url_for("login", next=request.path))
        return view_fn(*args, **kwargs)
    return wrapper

# PDF text extraction with OCR fallback
def extract_pdf_text(file_path: str) -> str:
    text = ""
    pdf_document = fitz.open(file_path)
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        page_text = page.get_text()
        if not page_text.strip():  # OCR if page is scanned
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            page_text = pytesseract.image_to_string(img)
        text += page_text + "\n"
    return text

# Simple translator via OpenAI chat
def translate_with_openai(text: str, target_language: str) -> str:
    if not text.strip():
        return ""
    prompt = f"Translate the following text to {target_language}. Keep meaning, tone, and formatting.\n\n{text}"
    resp = oai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a world-class translator."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return (resp.choices[0].message.content if resp.choices else "") or ""

# -------------------------------------------------------
# Routes: Public pages
# -------------------------------------------------------
@app.route("/")
def index():
    # Home page (beautiful Tailwind UI)
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

# -------------------------------------------------------
# Auth
# -------------------------------------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = (request.form.get("name") or "").strip()
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""

        if not name or not email or not password:
            flash("All fields are required.", "error")
            return redirect(url_for("register"))

        # ensure unique email
        if users_collection.find_one({"email": email}):
            flash("Email already registered. Please login.", "warning")
            return redirect(url_for("login"))

        users_collection.insert_one({
            "name": name,
            "email": email,
            "password_hash": generate_password_hash(password),
            "created_at": datetime.now(timezone.utc),
        })

        # auto-login
        user = users_collection.find_one({"email": email})
        session["user_id"] = str(user["_id"])
        session["user_name"] = user.get("name")
        session["user_email"] = user.get("email")

        flash("Welcome! Account created.", "success")
        return redirect(url_for("index"))

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        next_url = request.args.get("next") or url_for("index")

        user = users_collection.find_one({"email": email})
        if not user or not check_password_hash(user["password_hash"], password):
            flash("Invalid email or password.", "error")
            return redirect(url_for("login", next=next_url))

        session["user_id"] = str(user["_id"])
        session["user_name"] = user.get("name")
        session["user_email"] = user.get("email")
        flash("Logged in.", "success")
        return redirect(next_url)

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out.", "success")
    return redirect(url_for("index"))

# -------------------------------------------------------
# Meeting Notes (login required)
# -------------------------------------------------------
@app.route("/upload", methods=["POST"])
@login_required
def upload():
    if "audio" not in request.files:
        flash("No audio file part", "error")
        return redirect(url_for("index"))

    file = request.files["audio"]
    language = request.form.get("language", "auto")

    if file.filename == "":
        flash("No selected file", "error")
        return redirect(url_for("index"))

    if not (file and allowed_file(file.filename)):
        flash("File type not allowed", "error")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    file.save(save_path)

    try:
        audio_path = extract_audio_from_file(save_path)
        transcript = transcribe_audio(audio_path, language)
        if not transcript:
            transcript = "⚠️ Transcription failed or empty."

        summary = summarize_text(transcript, target_language=language)
        if not summary:
            summary = {
                "executive_summary": "⚠️ Failed to generate summary",
                "key_points": [],
                "action_items": [],
                "decisions": [],
                "sentiment": "Unknown"
            }

        note_doc = {
            "filename": filename,
            "converted_wav": os.path.basename(audio_path),
            "language": language,
            "transcript": transcript,
            "summary": summary,
            "created_at": datetime.now(timezone.utc),
            "owner_id": session.get("user_id"),
        }
        inserted = notes_collection.insert_one(note_doc)
        note_id = str(inserted.inserted_id)

        note_doc["_id"] = note_id
        return render_template("notes.html", result=note_doc)
    except Exception as e:
        flash(f"Processing failed: {e}", "error")
        return redirect(url_for("index"))

@app.route("/record", methods=["POST"])
@login_required
def record():
    if "audio" not in request.files:
        return jsonify({"error": "No audio data received"}), 400

    language = request.form.get("language", "auto")
    file = request.files["audio"]

    suffix = os.path.splitext(secure_filename(file.filename))[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        temp_path = tmp.name

    try:
        audio_path = extract_audio_from_file(temp_path)
        transcript = transcribe_audio(audio_path, language=language)
        if not transcript:
            transcript = "⚠️ Transcription failed or empty."

        summary = summarize_text(transcript, target_language=language)
        if not summary:
            summary = {
                "executive_summary": "⚠️ Failed to generate summary",
                "key_points": [],
                "action_items": [],
                "decisions": [],
                "sentiment": "Unknown"
            }

        note_doc = {
            "filename": os.path.basename(temp_path),
            "converted_wav": os.path.basename(audio_path),
            "language": language,
            "transcript": transcript,
            "summary": summary,
            "created_at": datetime.now(timezone.utc),
            "owner_id": session.get("user_id"),
        }
        inserted = notes_collection.insert_one(note_doc)
        note_id = str(inserted.inserted_id)

        redirect_url = url_for("view_note", note_id=note_id)
        return jsonify({"redirect_url": redirect_url})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/notes/<note_id>")
@login_required
def view_note(note_id):
    try:
        doc = notes_collection.find_one({"_id": ObjectId(note_id)})
        if not doc:
            return "Note not found", 404
        # ownership check
        if doc.get("owner_id") and doc["owner_id"] != session.get("user_id"):
            return "Forbidden", 403

        doc["_id"] = str(doc["_id"])
        return render_template("notes.html", result=doc)
    except Exception as e:
        return f"Error: {e}", 500

# -------------------------------------------------------
# Translator & Converters (Public)
# -------------------------------------------------------
@app.route("/translator", methods=["GET", "POST"])
def translator():
    user_input = ""
    translated_text = ""
    target_language = request.form.get("target_lang", "Urdu")

    if request.method == "POST":
        # Text translation
        if "message" in request.form and request.form["message"].strip():
            user_input = request.form["message"]
            try:
                translated_text = translate_with_openai(user_input, target_language)
            except Exception as e:
                flash(f"Translation failed: {e}", "error")
                translated_text = ""

            translations_collection.insert_one({
                "type": "text",
                "source_text": user_input,
                "translated_text": translated_text,
                "target_language": target_language,
                "created_at": datetime.now(timezone.utc),
                "owner_id": session.get("user_id")
            })

        # PDF translation
        elif "pdf_file" in request.files:
            pdf_file = request.files["pdf_file"]
            if pdf_file.filename != "":
                in_name = secure_filename(pdf_file.filename)
                in_path = os.path.join(app.config["UPLOAD_FOLDER"], in_name)
                os.makedirs(os.path.dirname(in_path), exist_ok=True)
                pdf_file.save(in_path)
                try:
                    pdf_text = extract_pdf_text(in_path)
                except Exception as e:
                    flash(f"Failed to read PDF: {e}", "error")
                    pdf_text = ""

                if not pdf_text.strip():
                    translated_text = "Could not extract text from the PDF."
                    user_input = ""
                else:
                    user_input = pdf_text
                    try:
                        translated_text = translate_with_openai(pdf_text[:15000], target_language)
                    except Exception as e:
                        flash(f"Translation failed: {e}", "error")
                        translated_text = ""

                translations_collection.insert_one({
                    "type": "pdf",
                    "filename": in_name,
                    "source_text": user_input,
                    "translated_text": translated_text,
                    "target_language": target_language,
                    "created_at": datetime.now(timezone.utc),
                    "owner_id": session.get("user_id")
                })

    return render_template(
        "translator.html",
        user_input=user_input,
        translated_text=translated_text,
        target_language=target_language,
    )

@app.route("/pdf-to-docx", methods=["POST"])
def pdf_to_docx():
    pdf_file = request.files.get("pdf_file")
    if not pdf_file or pdf_file.filename == "":
        flash("Please select a PDF file.", "error")
        return redirect(url_for("translator"))

    in_name = secure_filename(pdf_file.filename)
    in_path = os.path.join(app.config["UPLOAD_FOLDER"], in_name)
    os.makedirs(os.path.dirname(in_path), exist_ok=True)
    pdf_file.save(in_path)

    try:
        pdf_text = extract_pdf_text(in_path)
        out_path = os.path.splitext(in_path)[0] + ".docx"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        doc = Document()
        for line in pdf_text.splitlines():
            doc.add_paragraph(line)
        doc.save(out_path)

        if not os.path.isfile(out_path):
            flash("Failed to create DOCX file.", "error")
            return redirect(url_for("translator"))

        conversions_collection.insert_one({
            "type": "pdf-to-docx",
            "src_filename": in_name,
            "out_filename": os.path.basename(out_path),
            "created_at": datetime.now(timezone.utc),
            "owner_id": session.get("user_id")
        })

        return send_file(out_path, as_attachment=True, download_name=os.path.basename(out_path))
    except Exception as e:
        flash(f"PDF to DOCX failed: {e}", "error")
        return redirect(url_for("translator"))

@app.route("/docx-to-pdf", methods=["POST"])
def docx_to_pdf():
    docx_file = request.files.get("docx_file")
    if not docx_file or docx_file.filename == "":
        flash("Please select a DOCX file.", "error")
        return redirect(url_for("translator"))

    in_name = secure_filename(docx_file.filename)
    in_path = os.path.join(app.config["UPLOAD_FOLDER"], in_name)
    os.makedirs(os.path.dirname(in_path), exist_ok=True)
    docx_file.save(in_path)

    try:
        doc = Document(in_path)
        out_path = os.path.splitext(in_path)[0] + ".pdf"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        c = canvas.Canvas(out_path, pagesize=letter)
        width, height = letter
        left_margin = 40
        top_margin = height - 40
        line_height = 14

        text_obj = c.beginText(left_margin, top_margin)
        text_obj.setFont("Times-Roman", 12)

        y = top_margin
        for para in doc.paragraphs:
            lines = para.text.splitlines() if para.text else [""]
            for line in lines:
                if y < 40:
                    c.drawText(text_obj)
                    c.showPage()
                    text_obj = c.beginText(left_margin, top_margin)
                    text_obj.setFont("Times-Roman", 12)
                    y = top_margin
                text_obj.textLine(line)
                y -= line_height

        c.drawText(text_obj)
        c.save()

        if not os.path.isfile(out_path):
            flash("Failed to create PDF file.", "error")
            return redirect(url_for("translator"))

        conversions_collection.insert_one({
            "type": "docx-to-pdf",
            "src_filename": in_name,
            "out_filename": os.path.basename(out_path),
            "created_at": datetime.now(timezone.utc),
            "owner_id": session.get("user_id")
        })

        return send_file(out_path, as_attachment=True, download_name=os.path.basename(out_path))
    except Exception as e:
        flash(f"DOCX to PDF failed: {e}", "error")
        return redirect(url_for("translator"))

@app.route("/image-to-pdf", methods=["POST"])
def image_to_pdf():
    image_file = request.files.get("image_file")
    if not image_file or image_file.filename == "":
        flash("Please select an image file.", "error")
        return redirect(url_for("translator"))

    in_name = secure_filename(image_file.filename)
    in_path = os.path.join(app.config["UPLOAD_FOLDER"], in_name)
    os.makedirs(os.path.dirname(in_path), exist_ok=True)
    image_file.save(in_path)

    try:
        with Image.open(in_path) as im:
            if im.mode in ("RGBA", "P"):
                im = im.convert("RGB")
            out_path = os.path.splitext(in_path)[0] + ".pdf"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            im.save(out_path, "PDF", resolution=300)

        if not os.path.isfile(out_path):
            flash("Failed to create PDF file.", "error")
            return redirect(url_for("translator"))

        conversions_collection.insert_one({
            "type": "image-to-pdf",
            "src_filename": in_name,
            "out_filename": os.path.basename(out_path),
            "created_at": datetime.now(timezone.utc),
            "owner_id": session.get("user_id")
        })

        return send_file(out_path, as_attachment=True, download_name=os.path.basename(out_path))
    except Exception as e:
        flash(f"Image to PDF failed: {e}", "error")
        return redirect(url_for("translator"))

# -------------------------------------------------------
# Main
# -------------------------------------------------------
if __name__ == "__main__":
    # Disable reloader on Windows to avoid socket errors with MediaRecorder
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
