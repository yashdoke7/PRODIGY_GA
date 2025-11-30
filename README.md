# 🤖 Unified Generative AI Suite

A comprehensive implementation of 5 generative AI models with FastAPI backend and Streamlit frontend.

## 🎯 Models Implemented

1. **GPT-2 Text Generation** - Transformer-based language model with adjustable sampling
2. **Markov Chain Generator** - Statistical text generation with custom corpus training
3. **Stable Diffusion** - Text-to-image generation using latent diffusion
4. **Pix2Pix cGAN** - Image-to-image translation (facades dataset)
5. **Neural Style Transfer** - Artistic style application using VGG19

## 🚀 Features

- REST API endpoints for all models
- Interactive web UI with Streamlit
- Lazy model loading for efficiency
- Support for custom parameters
- Production-ready deployment structure

## 🛠️ Tech Stack

- **Backend**: FastAPI, Uvicorn
- **ML Frameworks**: PyTorch, Transformers, Diffusers
- **Frontend**: Streamlit
- **Models**: HuggingFace Hub, PyTorch Hub

## 📦 Installation

```
# Clone repository
git clone https://github.com/YOUR_USERNAME/PRODIGY_GA.git
cd PRODIGY_GA

# Install dependencies
pip install -r requirements.txt
```

## ▶️ Usage

**Start FastAPI Backend:**
```
uvicorn app.api.main:app --reload
```

**Launch Streamlit UI:**
```
streamlit run app/ui/streamlit_ui.py
```

**API Documentation:** http://localhost:8000/docs

## 📸 Screenshots

<img width="2466" height="986" alt="image" src="https://github.com/user-attachments/assets/19565323-7fa8-4bb7-8148-3eb168da5218" />

## 🎓 Project Structure

```
PRODIGY_GA/
├── app/
│   ├── api/          # FastAPI endpoints
│   ├── models/       # ML model implementations
│   └── ui/           # Streamlit interface
├── requirements.txt
└── README.md
```

## 🔗 Resources

- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- [Pix2Pix Paper](https://arxiv.org/abs/1611.07004)

## 📝 License

MIT License

## 👤 Author

**Your Name** - [LinkedIn](https://www.linkedin.com/in/yash-doke/) | [GitHub](https://github.com/yashdoke7)

Part of Prodigy Infotech Internship Program
```
