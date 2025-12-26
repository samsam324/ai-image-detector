# AI Image Detector

## 1) Put data in folders
- data/train/real/ -> real photos
- data/train/ai/ -> AI-generated images
- data/val/real/ -> real photos
- data/val/ai/ -> AI-generated images

## 2) Run backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

## 3) Run frontend
cd ../frontend
pip install -r requirements.txt
streamlit run app.py

## 4) Training
cd training
pip install -r requirements.txt
python extract_embeddings.py --split train
python extract_embeddings.py --split val
python train_classifier.py

After training, backend will automatically switch from demo_fallback to trained_model.
