import warnings
import os
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from PIL import Image
import io

app = FastAPI()

# Lazy loading globals
_gpt2 = None
_markov = None
_sd = None
_pix2pix = None
_style_transfer = None

def get_gpt2():
    global _gpt2
    if _gpt2 is None:
        from app.models.gpt2_text_generator import GPT2TextGenerator
        _gpt2 = GPT2TextGenerator()
    return _gpt2

def get_markov():
    global _markov
    if _markov is None:
        from app.models.markov_text_generator import MarkovTextGenerator
        _markov = MarkovTextGenerator()
    return _markov

def get_sd():
    global _sd
    if _sd is None:
        from app.models.stable_diffusion_generator import StableDiffusionGenerator
        _sd = StableDiffusionGenerator()
    return _sd

def get_pix2pix():
    global _pix2pix
    if _pix2pix is None:
        from app.models.pix2pix_img2img import Pix2Pix
        _pix2pix = Pix2Pix()
    return _pix2pix

def get_style_transfer():
    global _style_transfer
    if _style_transfer is None:
        from app.models.style_transfer import StyleTransfer
        _style_transfer = StyleTransfer()
    return _style_transfer

class MarkovTrainRequest(BaseModel):
    corpus: str

@app.get("/")
def root():
    return {"message": "Unified Generative AI API - All systems ready"}

@app.post("/generate_text_gpt2")
def generate_text_gpt2(prompt: str, max_length: int = 60, temperature: float = 0.9, 
                       top_k: int = 50, top_p: float = 0.95):
    model = get_gpt2()
    result = model.generate(prompt, max_length, temperature, top_k, top_p)
    return {"result": result}

@app.post("/train_markov")
def train_markov(req: MarkovTrainRequest):
    from app.utils.text_utils import clean_text
    model = get_markov()
    model.train(clean_text(req.corpus))
    return {"status": "trained"}

@app.post("/generate_text_markov")
def generate_text_markov(length: int = 50):
    model = get_markov()
    result = model.generate(length)
    return {"result": result}

@app.post("/generate_image")
def generate_image(prompt: str, steps: int = 50, scale: float = 7.5):
    model = get_sd()
    img = model.generate_image(prompt, num_inference_steps=steps, guidance_scale=scale)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

@app.post("/translate_image")
def translate_image(file: UploadFile = File(...)):
    model = get_pix2pix()
    image = Image.open(file.file)
    out_img = model.translate(image)
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

@app.post("/style_transfer")
def style_transfer_endpoint(content_file: UploadFile = File(...), 
                           style_file: UploadFile = File(...),
                           content_weight: float = 1e5, 
                           style_weight: float = 1e10, 
                           steps: int = 300):
    model = get_style_transfer()
    content_img = Image.open(content_file.file)
    style_img = Image.open(style_file.file)
    out_img = model.transfer(content_img, style_img, content_weight, style_weight, num_steps=steps)
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
