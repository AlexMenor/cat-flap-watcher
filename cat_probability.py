import cv2

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
