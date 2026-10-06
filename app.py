import os
import torch
import torch.nn as nn
import joblib
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from torchvision import transforms, models
from flask import Flask, render_template, request, redirect, jsonify
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DeterministicFakeEmbedding

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads/"

load_dotenv()
GROQ_API_KEY = os.getenv("API_KEY_GROQ")

# DL model
def load_dl_model():
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("pneumonia_resnet18.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

dl_model = load_dl_model()

dl_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.4456, 0.406], [0.229, 0.224, 0.225])
])

# ML Model
ml_model = joblib.load("patient_risk_model.pkl")

# RAG Setup
loader = PyPDFLoader("pneumonia_guide.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 400, chunk_overlap = 40)
docs = text_splitter.split_documents(documents)

embedding_model = DeterministicFakeEmbedding(size = 768)
vector_store = FAISS.from_documents(docs, embedding_model)

llm = ChatGroq(
    groq_api_key = GROQ_API_KEY,
    model = "qwen/qwen3.6-27b",
    temperature = 0.4,
    max_tokens=800
)

#FLASK Routes

@app.route("/", methods = ["GET","POST"])
def home():
    if request.method == "POST":
        age = int(request.form["age"])
        fever = float(request.form["fever"])
        oxygen = int(request.form["oxygen"])
        cough = int(request.form["cough"])
        breath = int(request.form["breath"])
        file = request.files["xray_image"]

        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        #layer 1 - image prediction
        img = Image.open(file_path).convert("RGB")
        img_tensor = dl_transforms(img).unsqueeze(0)
        with torch.no_grad():
            outputs = dl_model(img_tensor)
            _, preds = torch.max(outputs, 1)
            probabilities = torch.nn.functional.softmax(outputs, dim = 1)[0]

        dl_classes = ["NORMAL","PNEUMONIA"]
        dl_result = dl_classes[preds.item()]
        dl_confidence = f"{probabilities[preds.item()].item() * 100:.2f}%"

        #Layer 2:Risk Score[ML]
        input_data = np.array([[age, fever, oxygen, cough, breath]])
        ml_prediction = ml_model.predict(input_data)[0]
        ml_classes = ["Low Risk", "Medium Risk", "High Risk"]
        ml_result = ml_classes[ml_prediction]

        #Layer 3: RAG+Groq
        query = f"Pnemonia patient precautions treatment guidelines"
        matching_docs = vector_store.similarity_search(query, k = 2)
        context = "\n".join([doc.page_content for doc in matching_docs])

        prompt = f"""
        You are an expert AI Medical Assistant. Based on the following clinical data andmedical guidelines, generate a clear, professinal patient summary in easy-to-understand Hindi Language.
        Patient Clinical Data:
        - X-ray Diagnosis (DL Layer):{dl_result} (Confidence:{dl_confidence})
        - Symptom Risk Category (ML Layer):{ml_result}
        - Patient Age :{age} years 
        - Body Temperature:{fever} F
        - Oxygen Level (SpO2):{oxygen}%
        Verified Medical Guidelines (Context):{context}
         
        Instructions for Output:
        -Do NOT use any <think> tags.Do not output your thinking process.
        -output only the final responsein HIndi language directly.
        -Keep the response concise, stuructured, and under 300 words so it fits perfectly.
        - Write the response completely in Hindi language.
        - Start with a summary of the findings.
        - Provide 3-4 specific care steps or precautions based on the context.
        - End with a strict medical disclaimer that this is an AI tool and they must consult a doctor.
        """

        ai_response = llm.invoke(prompt)
        rag_summary = ai_response.content

        return jsonify({
            "dl_res":dl_result,
            "dl_conf":dl_confidence,
            "ml_res":ml_result,
            "rag_res":rag_summary
        })

    return render_template("index.html")

if __name__ == "__main__":
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok = True)
    app.run(debug=True)