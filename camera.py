import cv2
import datetime
import atexit
import requests
import os

cap = cv2.VideoCapture("http://192.168.8.105:6677/videofeed?username=&password=")

mog = cv2.createBackgroundSubtractorMOG2()

buffer_of_movements = []


def calculate_blur(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate the Laplacian variance as a measure of blur
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    return blur


def exit_handler():
    print("Cleanup function executed")
    cap.release()


atexit.register(exit_handler)

bot_token = os.environ.get("BOT_TOKEN")
chat_id = -1001975893303


def send_photo(photo_path):
    response = requests.post(
        f"https://api.telegram.org/bot{bot_token}/sendPhoto",
        data={"chat_id": chat_id},
        files={"photo": open(photo_path, "rb")},
    )
    if not response.ok:
        print("A request to telegram failed")


while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    fgmask = mog.apply(gray)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    fgmask = cv2.dilate(fgmask, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(
        fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    big_contours = [contour for contour in contours if cv2.contourArea(contour) > 1000]

    something_is_moving = len(big_contours) != 0

    if something_is_moving:
        buffer_of_movements.append(frame)
    elif not something_is_moving and len(buffer_of_movements) != 0:
        less_blurry = {"blur_score": float("inf"), "frame": None}
        for i in range(len(buffer_of_movements)):
            frame = buffer_of_movements[i]
            blur_score = calculate_blur(frame)
            if blur_score < less_blurry["blur_score"]:
                less_blurry = {"blur_score": blur_score, "frame": frame}
        photo_path = f"frame_{datetime.datetime.now().isoformat()}.jpg"
        cv2.imwrite(photo_path, frame)
        send_photo(photo_path)
        os.remove(photo_path)
        buffer_of_movements.clear()
