from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os
import cv2
import easyocr
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import io

app = FastAPI()

# Define directories for static files and symbols photos
static_dir = "static"
symbols_dir = "Symbols of Nutrition Photos"
text_records_dir = "text_records"
background_dir = "background"
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
if not os.path.exists(symbols_dir):
    os.makedirs(symbols_dir)
if not os.path.exists(text_records_dir):
    os.makedirs(text_records_dir)
if not os.path.exists(background_dir):
    os.makedirs(background_dir)

# Serve the static files (uploaded images), symbols photos, and text records
app.mount("/background", StaticFiles(directory=background_dir), name="background")
app.mount("/static", StaticFiles(directory=static_dir), name="static")
app.mount("/symbols", StaticFiles(directory=symbols_dir), name="symbols")
app.mount("/text_records", StaticFiles(directory=text_records_dir), name="text_records")

detected_texts_global = []

# Load the pre-trained model
model = load_model('my_model.h5')

# Class names for the model
class_names = ['CE', 'GF', 'greendot', 'OR', 'Period_After_Opening', 'Radura', 'Recycle', 'Tidyman', 'Yerli']

def remove_lines(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply adaptive thresholding
    bin_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10)
    # Detect horizontal and vertical lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detect_horizontal = cv2.morphologyEx(bin_image, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    detect_vertical = cv2.morphologyEx(bin_image, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    # Combine detected lines
    combined_lines = cv2.add(detect_horizontal, detect_vertical)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated_lines = cv2.dilate(combined_lines, kernel, iterations=3)
    lines_inv = cv2.bitwise_not(dilated_lines)
    # Inpaint the image to remove lines
    inpainted_image = cv2.inpaint(image, dilated_lines, 7, cv2.INPAINT_TELEA)
    return inpainted_image

def putTextWithCustomFont(img, text, position, font_path, font_scale=1, color=(255, 0, 0), thickness=1):
    # Convert OpenCV image to PIL image
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, int(font_scale * 30))

    # Draw the text
    draw.text(position, text, font=font, fill=color)

    # Convert back to OpenCV image
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img

def predict_image(img_path, model, class_names):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalization

    # Make a prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)

    # Map the predicted class index to the class name
    predicted_label = class_names[predicted_class[0]]
    return predicted_label

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return FileResponse('index.html')

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(static_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_url = f"/static/{file.filename}"
        return {"filename": file.filename, "content_type": file.content_type, "url": file_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {e}")

@app.post("/process_image")
async def process_image(data: dict):
    global detected_texts_global
    image_url = data['image_url']
    image_path = image_url.lstrip('/')

    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")

    try:
        img = cv2.imread(image_path)
        if img is None:
            raise HTTPException(status_code=400, detail="Failed to read image")

        img = remove_lines(img)
        reader = easyocr.Reader(['en','tr'], gpu=False)
        text = reader.readtext(img)
    
        threshold = 0.5
        detected_texts = []
        font_path = "./fonts/Arial.ttf" 
    
        for t in text:
            bbox, detected_text, score = t
            if score > threshold:
                detected_texts.append((bbox, detected_text))
    
        detected_texts_global = [detected_text for _, detected_text in detected_texts]

        # Sort detected texts by the top-left y-coordinate (to get top-to-bottom order)
        detected_texts.sort(key=lambda x: x[0][0][1])
    
        lines = []
        current_line = []
        current_y = detected_texts[0][0][0][1]  # Initial y-coordinate
    
        for bbox, detected_text in detected_texts:
            top_left_y = bbox[0][1]
            if abs(top_left_y - current_y) > 30:  # If y difference is greater than a threshold, start a new line
                lines.append(current_line)
                current_line = [detected_text]
                current_y = top_left_y
            else:
                current_line.append(detected_text)
    
        lines.append(current_line)  # Append the last line
    
        organized_text = "\n".join([" ".join(line) for line in lines])

        # Save the organized text to a file with UTF-8 encoding
        text_record_path = os.path.join(text_records_dir, os.path.splitext(os.path.basename(image_path))[0] + '.txt')
        with open(text_record_path, 'w', encoding='utf-8') as f:
            f.write(organized_text)

        for bbox, detected_text in detected_texts:
            cv2.rectangle(img, tuple(map(int, bbox[0])), tuple(map(int, bbox[2])), (0, 255, 0), 5)
            img = putTextWithCustomFont(img, detected_text, tuple(map(int, bbox[0])), font_path, font_scale=0.65, color=(255, 0, 0), thickness=1)

        processed_image_path = os.path.join(static_dir, 'processed_' + os.path.basename(image_path))
        cv2.imwrite(processed_image_path, img)
        processed_image_url = f"/static/{os.path.basename(processed_image_path)}"
    
        return {"processed_url": processed_image_url, "text_results": organized_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {e}")

@app.post("/predict_image")
async def predict_image_endpoint(data: dict):
    image_url = data['image_url']
    image_path = image_url.lstrip('/')

    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")

    try:
        predicted_label = predict_image(image_path, model, class_names)

        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))

        bbox_color = (0, 255, 0)  # Green color
        bbox_thickness = 2
        start_point = (10, 10)
        end_point = (210, 210)
        cv2.rectangle(img, start_point, end_point, bbox_color, bbox_thickness)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 255, 255)  # White color
        font_thickness = 1
        text_position = (20, 20)
        cv2.putText(img, predicted_label, text_position, font, font_scale, font_color, font_thickness)

        _, img_encoded = cv2.imencode('.png', img)
        img_bytes = img_encoded.tobytes()

        return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image prediction failed: {e}")

@app.get("/text_records")
async def get_text_records():
    try:
        text_files = [f for f in os.listdir(text_records_dir) if f.endswith('.txt')]
        text_contents = []

        for text_file in text_files:
            text_path = os.path.join(text_records_dir, text_file)
            with open(text_path, 'r', encoding='utf-8') as f:
                content = f.read()
                text_contents.append({"filename": text_file, "content": content})

        return text_contents
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve text records: {e}")

@app.post("/clear_files")
async def clear_files():
    global detected_texts_global
    detected_texts_global = []
    try:
        for directory in [static_dir, text_records_dir]:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Failed to delete {file_path}. Reason: {e}")
        return {"message": "All files have been cleared."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear files: {e}")

@app.get("/detected_texts")
async def get_detected_texts():
    try:
        return detected_texts_global
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve detected texts: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
