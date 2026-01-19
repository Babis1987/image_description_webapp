# Dockerfile for Hugging Face Spaces
FROM python:3.10-slim

# 1. Δημιουργία χρήστη με UID 1000 (απαραίτητο για HF Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# 2. Ορισμός working directory εντός του home του χρήστη
WORKDIR $HOME/app

# 3. Εγκατάσταση συστήματος (πρέπει να γίνει ως root)
USER root
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*
USER user

# 4. Αντιγραφή requirements και εγκατάσταση
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Αντιγραφή κώδικα με τα σωστά permissions
COPY --chown=user . .

# 6. Δημιουργία φακέλων (θα έχουν πλέον δικαιώματα εγγραφής)
RUN mkdir -p instance/uploads/original instance/uploads/processed instance/logs

# 7. Περιβάλλον
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV MODEL_TYPE=flan-t5
ENV USE_GPU=False
ENV USE_4BIT_QUANTIZATION=False

# 8. Πόρτα (Hugging Face uses 7860)
EXPOSE 7860

# 9. Εκτέλεση
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "600", "--workers", "1", "app:app"]