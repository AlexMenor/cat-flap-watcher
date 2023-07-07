import cv2
import atexit
import requests
import os
import datetime
import threading

with open("./coco.names", "rt") as f:
    classnames = f.read().rstrip("\n").split("\n")

config_path = "./ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weights_path = "./frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weights_path, config_path)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

threshold = 0.6
nms_threshold = 0.2


def calc_cat_probability(img):
    class_ids, confs, _ = net.detect(
        img, confThreshold=threshold, nmsThreshold=nms_threshold
    )
    classnames_to_look_for = ["cat", "dog", "horse", "sheep", "cow", "bear", "zebra"]
    max_probability = 0

    if len(class_ids) != 0:
        for class_id, confidence in zip(class_ids.flatten(), confs.flatten()):
            classname = classnames[class_id - 1]
            if classname in classnames_to_look_for:
                if confidence > max_probability:
                    max_probability = confidence

    return max_probability


def send_photo(photo_path, cat_probability):
    bot_token = os.environ.get("BOT_TOKEN")
    chat_id = -1001975893303
    response = requests.post(
        f"https://api.telegram.org/bot{bot_token}/sendPhoto",
        data={"chat_id": chat_id, "caption": f"Cat probability {cat_probability}%"},
        files={"photo": open(photo_path, "rb")},
    )
    if not response.ok:
        print(f"A request to telegram failed, status_code: {response.status_code}")


# bufferless VideoCapture
class VideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.lock = threading.Lock()
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True
        self.t.start()

    # grab frames as soon as they are available
    def _reader(self):
        while True:
            with self.lock:
                ret = self.cap.grab()
            if not ret:
                break

    # retrieve latest frame
    def read(self):
        with self.lock:
            _, frame = self.cap.retrieve()
        return frame

    def release(self):
        self.cap.release()


if __name__ == "__main__":
    cap = VideoCapture("http://192.168.8.108:6677/videofeed?username=&password=")

    def exit_handler():
        print("Cleanup function executed")
        cap.release()

    atexit.register(exit_handler)

    while True:
        frame = cap.read()
        cat_probability = calc_cat_probability(frame)

        if cat_probability != 0:
            photo_path = f"frame_{datetime.datetime.now().isoformat()}.jpg"
            cv2.imwrite(photo_path, frame)
            send_photo(photo_path, cat_probability)
            os.remove(photo_path)
