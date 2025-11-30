Install dependencies:
pip install -r requirements.txt

Run API server:
uvicorn app.api.main:app --reload

Run UI:
streamlit run app/ui/streamlit_app.py

Model Downloads:

First run will automatically download models (GPT-2, Stable Diffusion, VGG19, Pix2Pix).
For Pix2Pix with TensorFlow Hub: Install tensorflow_hub.