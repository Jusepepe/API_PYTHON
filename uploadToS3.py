import boto3
import os
import time
import sys
import threading

class ProgressPercentage:
    def __init__(self, filename):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        # To simplify, assume this is running in a single thread.
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            sys.stdout.write(
                "\r%s  %s / %s  (%.2f%%)" % (
                    self._filename, self._seen_so_far, self._size,
                    percentage)
            )
            sys.stdout.flush()

s3: boto3.client = boto3.client("s3")

day: str = time.strftime("%Y-%m-%d", time.localtime())
hour: str = time.strftime("%H:00_%p", time.localtime())


path: str = os.path.basename("./images")

track: str = "Track_" + str(1)

for i,file in enumerate(os.listdir(path)):
    s3.upload_file(Filename=os.path.join(path, file), Bucket="citric-bucket", Key=day + "/raw/" + hour + "/" + track + "/" + file, Callback=ProgressPercentage(os.path.join(path, file)))
    if i%10==0 and i!=0:
        track = "Track_" + str(int(track.split("_")[1]) + 1)

