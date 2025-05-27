# 🧠 Brain Tumor Classifier Backend

This is the **FastAPI backend** for the Brain Tumor Classifier project. It provides an API for classifying brain tumor images using a deep learning model. It also supports image uploading, prediction, and integration with a frontend application.

---

## 🔗 Repository

GitHub: [https://github.com/NITESH-SINGH-SE/brain_tumor_classifier_backend.git](https://github.com/NITESH-SINGH-SE/brain_tumor_classifier_backend.git)

---

## 🛠️ Features

- FastAPI-based RESTful backend
- Predicts brain tumor presence/type from uploaded images
- OpenAI integration
- Supports local and remote backend URLs
- Generates PDF reports (if applicable)

---

## 🧪 Setup Instructions

### ✅ Prerequisites

- Python 3.8+
- `pip` (Python package manager)
- (Optional) [Miniconda or Anaconda](https://docs.conda.io/en/latest/miniconda.html) for Conda-based environments

---

### 📦 Step 1: Clone the Repository

```bash
git clone https://github.com/NITESH-SINGH-SE/brain_tumor_classifier_backend.git
cd brain_tumor_classifier_backend
```

---

## 📁 Step 2: Environment Setup

### 🧰 Option A: Using Conda (Recommended)

If you are using Conda and want to replicate the exact environment:

1. Create the environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
```

2. Activate the environment:

```bash
conda activate tumor-env
```

> You can regenerate this file later using:  
> `conda env export --no-builds > environment.yml`

---

Then install dependencies:

```bash
pip install -r requirements.txt
```

---

### 🔐 Step 3: Environment Variables

Create a `.env` file in the root directory and add the following:

```env
OPENAI_API_KEY=your-openai-api-key
```

> Make sure not to commit `.env` to version control. It's ignored in `.gitignore`.

---

### 🚀 Step 4: Run the FastAPI Server

```bash
uvicorn main:app --reload
```

This will start the server at: [http://127.0.0.1:8000](http://127.0.0.1:8000)

Visit: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to see the Swagger UI.

---

## 📄 Files Included

- `main.py` — Main FastAPI app
- `requirements.txt` — pip dependencies
- `environment.yml` — Conda environment definition
- `.env` — Secrets like API keys (not included in repo)
- `README.md` — Setup instructions

---

## 🤝 Contributing

Feel free to open issues or pull requests for improvements or bug fixes.

---
