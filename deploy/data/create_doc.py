from flask import Flask, render_template_string, request, redirect, url_for, flash
from fpdf import FPDF
import os

app = Flask(__name__)
app.secret_key = "secret"  # Needed for flashing messages

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Patient PDF Generator</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f7f7f7; margin: 0; padding: 0; }
        .container { max-width: 600px; margin: 40px auto; background: #fff; border-radius: 8px; box-shadow: 0 2px 8px #ccc; padding: 32px; }
        h2 { text-align: center; color: #333; }
        textarea { width: 100%; min-height: 180px; font-size: 15px; font-family: 'Courier New', Courier, monospace; padding: 10px; border-radius: 4px; border: 1px solid #bbb; margin-bottom: 18px; }
        button { background: #007bff; color: #fff; border: none; padding: 12px 32px; border-radius: 4px; font-size: 16px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .msg { margin: 10px 0; color: green; text-align: center; }
        .err { margin: 10px 0; color: red; text-align: center; }
    </style>
</head>
<body>
    <div class="container">
        <h2>Patient PDF Generator</h2>
        {% with messages = get_flashed_messages(with_categories=true) %}
          {% if messages %}
            {% for category, message in messages %}
              <div class="{{ category }}">{{ message }}</div>
            {% endfor %}
          {% endif %}
        {% endwith %}
        <form method="post">
            <label for="details"><b>Enter patient details (name, age, etc.):</b></label><br>
            <textarea id="details" name="details" placeholder="Name: John Doe&#10;Age: 45&#10;Diagnosis: ..."></textarea><br>
            <button type="submit">Create PDF</button>
        </form>
    </div>
</body>
</html>
"""

def replace_unicode_punctuation(text):
    # Replace common unicode punctuation with ASCII equivalents
    replacements = {
        "\u2018": "'",  # left single quotation mark
        "\u2019": "'",  # right single quotation mark
        "\u201c": '"',  # left double quotation mark
        "\u201d": '"',  # right double quotation mark
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2026": "...",  # ellipsis
        "\u00a0": " ",  # non-breaking space
    }
    for uni, ascii_ in replacements.items():
        text = text.replace(uni, ascii_)
    return text

def safe_ascii(text):
    # Replace unicode punctuation, then encode to ascii, replacing unknowns with '?'
    text = replace_unicode_punctuation(text)
    return text.encode("latin-1", "replace").decode("latin-1")

def generate_pdf(patient_details):
    lines = patient_details.strip().split("\n")
    if not lines or ":" not in lines[0]:
        raise ValueError("First line must contain 'Name: ...'")
    patient_name = lines[0].split(":", 1)[1].strip()  # Extracting name from first line

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for line in lines:
        # Convert line to safe ASCII for FPDF (which uses latin-1)
        safe_line = safe_ascii(line)
        pdf.cell(200, 10, txt=safe_line, ln=True, align="L")

    # Ensure ./document/new directory exists
    save_dir = "./document/new"
    os.makedirs(save_dir, exist_ok=True)

    # Save PDF
    safe_name = "".join(c for c in patient_name if c.isalnum() or c in (' ', '_', '-')).rstrip()
    pdf_file_path = os.path.join(save_dir, f"{safe_name}.pdf")
    pdf.output(pdf_file_path)
    return pdf_file_path

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        details = request.form.get("details", "").strip()
        if not details:
            flash("No input provided. Please enter patient details.", "err")
            return redirect(url_for("index"))
        try:
            pdf_path = generate_pdf(details)
            flash(f"PDF saved successfully: {pdf_path}", "msg")
        except Exception as e:
            # If it's a UnicodeEncodeError, give a more helpful message
            if isinstance(e, UnicodeEncodeError):
                flash("Error generating PDF: Some characters could not be encoded. Please avoid using special punctuation or non-English characters.", "err")
            else:
                flash(f"Error generating PDF: {e}", "err")
        return redirect(url_for("index"))
    return render_template_string(HTML_TEMPLATE)

if __name__ == "__main__":
    app.run(debug=True)