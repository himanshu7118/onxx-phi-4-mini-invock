# 🧠 Phi-4-Mini Intent-Aware Assistant (ONNX + Streamlit)

An intelligent AI assistant powered by Microsoft's `Phi-4-mini-instruct` model. This assistant is optimized for local inference using **ONNX Runtime GenAI** and is capable of understanding user **intent**, extracting **context**, and providing accurate, structured responses.

---

## 📌 Features

- ✅ Local model inference (no API key required)
- 🧠 Intent and context extraction
- ⚡ Low-latency response generation
- 🪄 ONNX Runtime GenAI acceleration
- 🖥️ Clean, interactive UI with Streamlit
- 🧩 JSON-formatted answers for easy integration

---

## 🛠️ Setup Instructions

Follow these steps to get everything running locally.

---

### steps to follow: 

1. 📁 Create Your Project Directory

```bash
mkdir phi4-intent-assistant
cd phi4-intent-assistant
```

2. 🧠 Load and Save the Hugging Face Model
Save the following as download_model.py:

```bash
from transformers import AutoTokenizer, AutoModelForCausalLM, logging

# Enable logging
logging.set_verbosity_info()

# Save path
local_dir = "./phi4-mini"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct", trust_remote_code=True)
tokenizer.save_pretrained(local_dir)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-4-mini-instruct", trust_remote_code=True)
model.save_pretrained(local_dir)

print("Model and tokenizer saved to", local_dir)
```

Then run:
```bash
python model.py
```

3. 📦 Install Required Libraries

```bash
pip install olive-ai
pip install transformers
pip install --pre onnxruntime-genai
pip install streamlit
```

4. ⚙️ Optimize the Model Using Olive
This command will convert and optimize the model using ONNX Runtime GenAI:

```bash
olive auto-opt \
  --model_name_or_path ./phi4-mini \
  --output_path ./phi4-mini-optimized \
  --device cpu \
  --provider CPUExecutionProvider \
  --precision int4 \
  --use_model_builder \
  --log_level 1
```

5. 🚀 Run the App

```bash
streamlit run app.py
```

Then open your browser and go to http://localhost:8501.

### 🧠 Example Output
<|user|> What are some beginner projects in Python?
</s>
<|assistant|> 
{
  "intent": "User is seeking beginner project ideas for learning Python.",
  "context": "Likely a new learner looking for hands-on coding experience.",
  "response": "You can start with a to-do list app, a simple calculator, or a weather info CLI tool using an API like OpenWeather."
}

### 🧠 Tech Stack
microsoft/Phi-4-mini-instruct

ONNX Runtime GenAI

Olive AI Optimizer

Streamlit

Python 3.10+

### 📄 License
This project is licensed under the MIT License © 2025 Himanshu Rami
