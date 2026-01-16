from flask import Flask, render_template, request, send_from_directory, session, redirect, url_for
from config import ORIGINAL_DIR, PROCESSED_DIR, SECRET_KEY, allowed_file
from shutil import copyfile
from core.pipeline import process_image
from core.description_generator import DescriptionGenerator
from datetime import datetime 
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import cv2


    


app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY

def process_one_file(file, model_choice):
    filename = file.filename or ""
    if not filename:
        return None, "Empty filename"

    save_path = ORIGINAL_DIR / filename
    file.save(save_path)

    image = cv2.imread(str(save_path))
    if image is None:
        return None, f"Not a valid image (broken or unsupported format)"

    try:
        try:
            result = run_pipeline_with_timeout(image, timeout_sec=600)  # 10 λεπτά = 600 sec
        except TimeoutError:
            return None, "Timeout: processing exceeded 10 minutes"
    except Exception as e:
        return None, f"Processing error ({type(e).__name__})"

    processed_path = PROCESSED_DIR / filename
    annotated = result.get("annotated_image") if result.get("annotated_image") is not None else image
    cv2.imwrite(str(processed_path), annotated)

    processed_url = f"/uploads/processed/{filename}"
    description = (result.get("description") or "").strip()
    if not description:
        # empty response handling
        detections = result.get("faces_detected") or 0
        description = "No faces detected in the image." if not detections else "No description generated (empty model output)."

    return {
        "filename": filename,
        "processed_url": processed_url,
        "description": description,
        "model": model_choice
    }, None

def run_pipeline_with_timeout(image, timeout_sec: int):
    future = EXECUTOR.submit(
        process_image,
        image=image,
        generator=GEN,
        generate_description=True,
        visualize=True
    )
    return future.result(timeout=timeout_sec)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["GET", "POST"])
def chat():
    if request.method == "POST":

        model_choice = (request.form.get("model") or "mistral").lower()
        if model_choice not in ("mistral", "flan-t5"):
            model_choice = "mistral"
        GEN.switch_model(model_choice)
        
        session.setdefault("all_results", [])
        already = {
            (r.get("filename"), r.get("model"))
            for r in session["all_results"]
        }

        files = request.files.getlist("images")
        
        skipped = []
        results = []
        for file in files:

            key = (file.filename, model_choice)
            if key in already:
                if key in already:
                    skipped.append(f"{file.filename} (already analyzed with {model_choice})")
                    continue
            item, reason = process_one_file(file, model_choice)
            
            if reason:
                skipped.append(f"{file.filename} ({reason})")
                continue

            
            item['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            results.append(item)
            already.add((item["filename"], item["model"]))
           

        if 'all_results' not in session:
            session['all_results'] = []

        session['all_results'].extend(results)  # Πρόσθεσε τα νέα στη λίστα
        session.modified = True  # Σημαντικό για nested objects!

        message = f"Processed {len(results)} image(s)."
        if skipped:
            shown = "; ".join(skipped)
            message += f" Skipped {len(skipped)} file(s): {shown}."

        session['message'] = message
        session['model_choice'] = model_choice

        return redirect(url_for('chat'))
     
    message = session.pop('message', None)
    model_choice = session.get('model_choice', 'mistral')
    return render_template(
        "chat.html", 
        model_choice=model_choice, 
        results=session.get('all_results', [])[::-1],  # ⬅️ Reverse για νεότερα πρώτα
        message=message
    )


@app.route("/start-new-session", methods=["POST"])
def start_new_session():
    session.pop("all_results", None)
    session["message"] = "Started a new session."
    return redirect(url_for("chat"))


@app.route("/instructions")
def instructions():
    return render_template("instructions.html")

@app.route("/uploads/original/<path:filename>")
def uploaded_original(filename):
    return send_from_directory(ORIGINAL_DIR, filename)

@app.route("/uploads/processed/<path:filename>")
def uploaded_processed(filename):
    return send_from_directory(PROCESSED_DIR, filename)

@app.route("/clear", methods=["POST"])
def clear_history():
    session.pop('all_results', None)
    return redirect(url_for("chat"))

# Initialize models at module level for gunicorn
GEN = DescriptionGenerator(model_type="flan-t5", lazy_load=True)
EXECUTOR = ThreadPoolExecutor(max_workers=1)

if __name__=="__main__":
    app.run(debug=True, host="0.0.0.0", port=7860)
