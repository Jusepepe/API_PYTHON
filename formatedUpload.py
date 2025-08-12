import boto3
import time
import os
import io

s3: boto3.client = boto3.client("s3")

bucket_name: str = "citric-bucket"

day: str = time.strftime("%Y-%m-%d", time.localtime())
hour: str = time.strftime("%H:00_%p", time.localtime())
track: str = "Track_" + str(1)

views: dict = {
    "front": {
        "pan": ["left","center","right"],
        "tilt": [0, 90, 180]
    },
    "back": {
        "pan": ["left","center","right"],
        "tilt": [0, 90, 180]
    }
}



for view in views:
    for pan in views[view]["pan"]:
        print(view, pan)
