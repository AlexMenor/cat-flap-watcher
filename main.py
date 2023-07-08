import cv2
from queue import Queue
import atexit
import requests
import os
import datetime
import threading
from cat_probability import calc_cat_probability

bot_token = os.environ.get("BOT_TOKEN")
chat_id = -1001975893303


def consumer(queue):
    print("Consumer: Running")
    while True:
        item = queue.get()
        _, img_encoded = cv2.imencode(".jpg", item["frame"])
        print(f"Notifying about cat with probability {item['cat_probability']}")
        response = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendPhoto",
            data={
                "chat_id": chat_id,
                "caption": f"Cat probability {round(item['cat_probability'] * 100,2)}%",
            },
            files={"photo": img_encoded.tobytes()},
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
    cap = VideoCapture("http://192.168.8.109:6677/videofeed?username=&password=")

    queue = Queue(20)

    threading.Thread(target=consumer, args=(queue,), daemon=True).start()

    def exit_handler():
        print("Cleanup function executed")
        cap.release()

    atexit.register(exit_handler)

    event_ongoing_timestamp = None
    event_items = []

    while True:
        frame = cap.read()
        cat_probability = calc_cat_probability(frame)
        current_timestamp = datetime.datetime.now()

        if cat_probability != 0:
            item = {
                "timestamp": current_timestamp,
                "frame": frame,
                "cat_probability": cat_probability,
            }
            event_items.append(item)
            print(f"Appending item to the event with probability {cat_probability}")
            if event_ongoing_timestamp is None:
                event_ongoing_timestamp = current_timestamp
                print(f"Event started at ${event_ongoing_timestamp.isoformat()}")

        if (
            event_ongoing_timestamp is not None
            and (current_timestamp - event_ongoing_timestamp).total_seconds() > 20
        ):
            print(f"Event finished at ${current_timestamp.isoformat()}")
            best_item = max(event_items, key=lambda x: x["cat_probability"])
            queue.put_nowait(best_item)
            event_ongoing_timestamp = None
            event_items.clear()
