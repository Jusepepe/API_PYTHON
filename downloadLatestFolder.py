import boto3
import time
from ultralytics import YOLO
from PIL import Image
import io

model = YOLO("./my_model/my_model.pt")

s3: boto3.client = boto3.client("s3")

day: str = time.strftime("%Y-%m-%d", time.localtime())
hour: str = time.strftime("%H:00_%p", time.localtime())

directories: dict = s3.list_objects_v2(Bucket="citric-bucket", Prefix=day+"/raw/"+hour+"/")

objects: list = directories.get("Contents")

for object in objects:
    key: str = object.get("Key")
    print(key, ":")
    buffer0: io.BytesIO = io.BytesIO()
    s3.download_fileobj(Bucket="citric-bucket", Key=key, Fileobj=buffer0)
    buffer0.seek(0)
    img: Image.Image = Image.open(buffer0)
    results: list = model([img])
    result: Image.Image = results[0].plot()
    im_rgb: Image.Image = Image.fromarray(result[..., ::-1])
    buffer1: io.BytesIO = io.BytesIO()
    im_rgb.save(buffer1, format="JPEG")
    buffer1.seek(0)
    s3.upload_fileobj(
        Bucket="citric-bucket",
        Key=key.replace("raw", "processed"),
        Fileobj=buffer1
    )
    print("Processed")
    


