from flask import Flask, send_file, request
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)
model = YOLO("./my_model/my_model.pt")
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000

@app.post("/image")
def pest_detection():
    
    file = request.files['file']
    img = Image.open(file)
    result = model([img])  # results list
    im_bgr = result[0].plot()
    im_rgb = Image.fromarray(im_bgr[..., ::-1]) 

    # Guardar en un buffer en memoria
    buffer = io.BytesIO()
    im_rgb.save(buffer, format="JPEG")
    buffer.seek(0)

    # Retornar la imagen como respuesta HTTP
    return send_file(buffer, mimetype='image/jpeg', download_name="result.jpg")

if __name__ == "__main__":
    app.run(port=80, debug=True)