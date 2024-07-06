from flask import Flask, request, jsonify
from ultralytics import YOLO
import os
import subprocess

# WEIGHT_PATH = os.environ["WEIGHTS_PATH"]
# VIDEO_RESULT_PATH = os.environ["VIDEO_RESULT_PATH"]
WEIGHT_PATH="./yolo/yolov8n.pt"
VIDEO_RESULT_PATH="/opt/project/data/result_video"
app = Flask(__name__)

configMap = {
    "conf": 0.6, 
    "iou": 0.5,
}
names = {0: 'person',
 1: 'bicycle',
 2: 'car',
 3: 'motorcycle',
 4: 'airplane',
 5: 'bus',
 6: 'train',
 7: 'truck',
 8: 'boat',
 9: 'traffic light',
 10: 'fire hydrant',
 11: 'stop sign',
 12: 'parking meter',
 13: 'bench',
 14: 'bird',
 15: 'cat',
 16: 'dog',
 17: 'horse',
 18: 'sheep',
 19: 'cow',
 20: 'elephant',
 21: 'bear',
 22: 'zebra',
 23: 'giraffe',
 24: 'backpack',
 25: 'umbrella',
 26: 'handbag',
 27: 'tie',
 28: 'suitcase',
 29: 'frisbee',
 30: 'skis',
 31: 'snowboard',
 32: 'sports ball',
 33: 'kite',
 34: 'baseball bat',
 35: 'baseball glove',
 36: 'skateboard',
 37: 'surfboard',
 38: 'tennis racket',
 39: 'bottle',
 40: 'wine glass',
 41: 'cup',
 42: 'fork',
 43: 'knife',
 44: 'spoon',
 45: 'bowl',
 46: 'banana',
 47: 'apple',
 48: 'sandwich',
 49: 'orange',
 50: 'broccoli',
 51: 'carrot',
 52: 'hot dog',
 53: 'pizza',
 54: 'donut',
 55: 'cake',
 56: 'chair',
 57: 'couch',
 58: 'potted plant',
 59: 'bed',
 60: 'dining table',
 61: 'toilet',
 62: 'tv',
 63: 'laptop',
 64: 'mouse',
 65: 'remote',
 66: 'keyboard',
 67: 'cell phone',
 68: 'microwave',
 69: 'oven',
 70: 'toaster',
 71: 'sink',
 72: 'refrigerator',
 73: 'book',
 74: 'clock',
 75: 'vase',
 76: 'scissors',
 77: 'teddy bear',
 78: 'hair drier',
 79: 'toothbrush'}

yolo = YOLO(model=WEIGHT_PATH,verbose=False)

def convert_avi_to_mp4(avi_file_path, output_name):
    cmd = "ffmpeg -i '{input}' -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 '{output}' && rm -r {track}".format(input = avi_file_path, output = output_name, track = '/'.join(avi_file_path.split("/")[0:-1]))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    (output, err) = p.communicate()  
    p.wait()
    return True


@app.route("/get-predict", methods=['POST'])
def hello_world():
    """
    ret:
        return dict like: 
        {
            "frame_0":[[
                x1,
                y1,
                x2,
                y2,
                object_id,
                confidience,
                class
                ],
                [
                x1,
                y1,
                x2,
                y2,
                object_id,
                confidience,
                class
                ],
                -- // --
            ],
            -- // --
        }
    """
    req = request.get_json()
    res = yolo.predict(source=req['file_path'], project=VIDEO_RESULT_PATH, **configMap)
    #track(source=req['file_path'],save=True,project=VIDEO_RESULT_PATH, **configMap)
    result_dict = {}
    for r_ind, r in enumerate(res):
        result = r.boxes.data.cpu().tolist()
        for i in range(len(result)):
            result[i][-1]= names[result[i][-1]]
        result_dict[f"frame_{r_ind}"] = result

    file_name = req['file_path'].split("/")[-1].split(".")[0]
    result_dict["res_file_path"] = VIDEO_RESULT_PATH + "/result_" + file_name + ".mp4"

    convert_avi_to_mp4(VIDEO_RESULT_PATH+"/track/"+file_name+".avi", result_dict["res_file_path"])
    return jsonify(result_dict), 200

@app.route("/yolo/health/live")
def health():
    return "", 200