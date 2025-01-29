from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load YOLO model (replace the path with your model's location)
damage_model = YOLO('/Users/adoomy/GP2_Final/Auto_final/best.pt')
damage_class_names = damage_model.names

# Load Llama3.1 (Meta's Llama-3.1-8B-Instruct model)
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16  # Use mixed-precision computation
).to("cuda")  # Ensure the model runs on GPU

app = FastAPI()

# Pre-tokenize the static part of the prompt
static_prompt = """
Role and Context:
"You are an expert automotive analyst specializing in generating detailed and structured car damage analysis reports. Your task is to analyze a given car image and generate a detailed, human-readable damage analysis report."
### Context ###
The car image contains visible damages that need to be described comprehensively.
You are required to break down the analysis into clear and concise sections.
### Instructions ###
Analyze the car image and generate a detailed damage analysis report in a structured format.
Ensure the report includes the following sections:
Car Damage Analysis Report
Detected Damages: List all detected damages with the following details:
Type of damage.
Bounding box coordinates.
Confidence scores for detection.
Step-by-Step Analysis
Overview of Car Image Analysis: Provide a brief explanation of the method and purpose of the analysis.
Summary of Detected Damages: Offer a concise summary highlighting the type and locations of damages observed.
Tone and Style
Write the report in professional and clear language.
Avoid technical jargon that might be unfamiliar to general readers.
Ensure the content is concise yet descriptive.
Output Format
Write the report as plain text without including code.
"""
static_inputs = tokenizer(static_prompt, return_tensors="pt")  # Cache the tokenized static prompt
static_inputs["input_ids"] = static_inputs["input_ids"].to("cuda")
static_inputs["attention_mask"] = static_inputs["attention_mask"].to("cuda")

@app.post("/analyze_image/")
async def analyze_image(file: UploadFile = File(...)):
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG and PNG are supported.")

    # Read image data
    image_data = await file.read()
    image = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Enhance image quality
    def enhance_image(img):
        # Convert to grayscale for better contrast adjustment
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply histogram equalization to improve contrast
        gray = cv2.equalizeHist(gray)
        # Convert back to BGR
        enhanced_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # Apply GaussianBlur to reduce noise
        enhanced_img = cv2.GaussianBlur(enhanced_img, (5, 5), 0)
        # Apply sharpening kernel
        sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        enhanced_img = cv2.filter2D(enhanced_img, -1, sharpening_kernel)
        return enhanced_img

    image = enhance_image(image)

    # Detect damages using YOLO
    results = damage_model(image)

    # Process YOLO detections
    damages = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        conf = float(box.conf[0])              # Confidence score
        cls = int(box.cls[0])                  # Class index

        damages.append({
            "bounding_box": [x1, y1, x2 - x1, y2 - y1],  # Convert to (x, y, w, h)
            "confidence": conf,
            "class_name": damage_class_names[cls]
        })

    # Add detected damages to the prompt
    dynamic_prompt = ""
    if damages:  # Check if any damages were found
        for damage in damages:
            dynamic_prompt += f"\n- {damage['class_name']} at coordinates [{', '.join(map(str, damage['bounding_box']))}] with a confidence score of {damage['confidence']:.2f}"
    else:
        dynamic_prompt += "\n No damages were detected."

    # Combine static and dynamic parts of the prompt
    combined_prompt = static_prompt + dynamic_prompt
    inputs = tokenizer(combined_prompt, return_tensors="pt").to("cuda")

    # Generate text with the LLM
    output = model.generate(
        **inputs,
        max_length=800, 
        no_repeat_ngram_size=3,  # Avoid repetitive sentences
        early_stopping=True,
    )

    # Decode the generated text
    full_analysis = tokenizer.decode(output[0], skip_special_tokens=True).strip()

    # Remove the original prompt from the generated text
    final_report = full_analysis.replace(static_prompt.strip(), "").strip()

    return JSONResponse(content={
        "damages": damages,
        "analysis": final_report  # Return only the final report
    })
