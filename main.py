import os
import time
import shapely
from shapely.geometry import Polygon
import pickle5 as pickle
from mrcnn.model import MaskRCNN
import mrcnn.utils
import mrcnn.config
import cv2
import numpy as np
from flask import Flask, Response
from time import ctime

print(cv2.__version__)
print(cv2.getBuildInformation())


# Инициализирую класс видеопотока
class ParkingDetector:
    # здесь подается ссылка на rtsp поток
    def __init__(self):
        video_source = "Введите свой RTSP поток"
        rtsp_latency = 20
        g_stream = f"rtspsrc location={video_source} latency={rtsp_latency} ! decodebin ! videoconvert ! appsink"
        start = time.time()
        self.video = cv2.VideoCapture(g_stream, cv2.CAP_GSTREAMER)
        self.success, self.frame = self.video.read()
        self.start_row = int(250)
        self.start_col = int(90)
        self.end_row = int(650)
        self.end_col = int(970)
        self.cropped_frame = self.frame[self.start_row:self.end_row, self.start_col:self.end_col]
        self.tracker = 12
        self.target = 12
        self.tmp = np.array([])
        self.time_list = []
        self.time_list.append(start)
        if not self.video.isOpened():
            print("Could not open feed")

    def __del__(self):
        self.video.release()

    # читаем видео, уменьшаем его и детектим. фун-я flow берет первый кадр, отсчитывает итерации и берет следующий кадр
    def get_frame(self):
        self.success, self.frame = self.video.read()
        while self.success:
            self.frame = cv2.resize(self.frame, (int(self.frame.shape[1]//2.4), int(self.frame.shape[0]//2.4)))
            self.cropped_frame = self.frame[self.start_row:self.end_row, self.start_col:self.end_col]
            start_time = time.time()
            self.time_list.append(start_time)
            #             print(f"Time: {ctime(start_time)} FPS : {fps}")
            return cv2.imencode('.jpg', self.flow())[1].tobytes()
        #             return cv2.imencode('.jpg', self.frame)[1].tobytes()

    # функция, которая берет первый попавшийся кадр, прогоняет его через детекцию, а потом сохраняет его в озу и
    # показывает дальше, когда переменная tracker итерируется
    def flow(self):
        if self.tracker == self.target:
            self.cropped_frame = detect(self.cropped_frame)
            count_truck, count_car, count_person, free_space = self.cropped_frame[1:5]
            self.cropped_frame = self.cropped_frame[0]
            self.frame = count_to_pic(self.frame, count_truck, count_car, count_person, free_space)
            height, width, depth = self.cropped_frame.shape
            self.frame[self.start_row: self.start_row + height,
            self.start_col: self.start_col + width] = self.cropped_frame
            start_time = time.time()
            self.time_list.append(start_time)
            fps = 1/(self.time_list[-1] - self.time_list[-2])
            self.tmp = self.frame
            self.tracker = 0
            print(f"Time: {ctime(start_time)} FPS : {fps}")
            return self.frame
        else:
            self.tracker += 1
            return self.tmp


def random_colors(num_classes):
    np.random.seed(13)
    colors = [tuple(255*np.random.rand(3)) for _ in range(num_classes)]
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n]*(1 - alpha) + alpha*c,
            image[:, :, n]
        )
    return image


def display_instances(image, boxes, masks, ids, names, scores):
    n_instances = boxes.shape[0]

    for i in range(n_instances):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        color = class_dict[label]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]
        if ids[i] in [3, 8, 6, 1, 4, 9]:
            image = apply_mask(image, mask, color)
            image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
            image = cv2.putText(image, caption, (x1, y1),
                                cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 1)
    return np.array(image)


# функция, которая вытаскивает машины из общего числа объектов
def get_cars(boxes, class_ids):
    cars = []
    for i, box in enumerate(boxes):
        if class_ids[i] in [1, 3, 8, 6]:
            cars.append(box)
    return np.array(cars)


# функция расчета занятого места
def compute_overlaps(dict_parked_car_boxes, car_boxes, dict_neighbors):
    new_car_boxes = []
    for box in car_boxes:
        p1 = (box[1], box[0])
        p2 = (box[3], box[0])
        p3 = (box[3], box[2])
        p4 = (box[1], box[2])
        new_car_boxes.append([p1, p2, p3, p4])

    overlaps = np.zeros((len(dict_parked_car_boxes), len(new_car_boxes)))
    bad_car_places_id = np.array([])
    for i in range(len(dict_parked_car_boxes)):
        neighbors = dict_neighbors[i]
        for j in range(car_boxes.shape[0]):
            polygon_1 = Polygon(dict_parked_car_boxes[i])
            polygon_2 = Polygon(new_car_boxes[j])

            intersection = polygon_1.intersection(polygon_2).area
            union = polygon_1.union(polygon_2).area
            IOU = intersection/union
            for neighbor in neighbors:
                polygon_neighbor = Polygon(dict_parked_car_boxes[neighbor])
                intersection_neighbor = polygon_neighbor.intersection(polygon_2).area
                union_neighbor = polygon_neighbor.intersection(polygon_2).area
                try:
                    IOU_neighbor = intersection_neighbor/union_neighbor
                    if IOU_neighbor != 0:
                        bad_car_places_id = np.append(bad_car_places_id, neighbor)
                        overlaps[i][j] = IOU
                except:
                    overlaps[i][j] = IOU

    return np.array(overlaps), bad_car_places_id


# функция обнаружения. Берется фрейм и на его основании детектятся тачки, рассчитываются свободные места

def detect(frame):
    results = model.detect([frame], verbose=1)
    # results = model.detect([frame], verbose=0)
    cars = get_cars(results[0]['rois'], results[0]['class_ids'])
    overlaps, bad_car_places_id = compute_overlaps(dict_parked_car_boxes, cars, dict_neighbors)

    """ 
    Говнокод
    """
    busy_car_places_id = np.array([])
    destroy_car_places_id = np.array([])

    free_space = 0
    bad_cars_places_count = 0

    # cv2.fillPoly(frame, bad_car_places, (235, 51, 35))
    try:
        for parking_id, overlap_areas in zip(dict_parked_car_boxes.keys(), overlaps):
            max_IoU_overlap = np.max(overlap_areas)
            if max_IoU_overlap < 0.05:
                cv2.fillPoly(frame, [np.array(dict_parked_car_boxes[parking_id])], (120, 200, 132))
                free_space += 1
            if max_IoU_overlap >= 0.25:
                busy_car_places_id = np.append(busy_car_places_id, parking_id)
    except:
        pass

    busy_car_places_id = np.unique(busy_car_places_id)
    busy_car_places_id = np.sort(busy_car_places_id)
    busy_car_places_id = busy_car_places_id.astype(np.int32)
    busy_car_places_id_list = []
    for number in busy_car_places_id:
        busy_car_places_id_list.append(number)
    bad_car_places_id = np.unique(bad_car_places_id)
    bad_car_places_id = np.sort(bad_car_places_id)
    bad_car_places_id = bad_car_places_id.astype(np.int32)
    bad_car_places_id_list = []
    for number in bad_car_places_id:
        bad_car_places_id_list.append(number)

    def filter_duplicate(string_to_check):
        if string_to_check in busy_car_places_id_list:
            return False
        else:
            return True

    not_normal_car_places_id = list(filter(filter_duplicate, bad_car_places_id_list))
    for place_id in not_normal_car_places_id:
        cv2.fillPoly(frame, [np.array(dict_parked_car_boxes[place_id])], (235, 51, 35))

    # if len(busy_car_places_id) >= len(bad_car_places_id):
    #     for i in range(0, len(busy_car_places_id)):
    #         for j in range(0, len(bad_car_places_id))

    # normal_car_places_id = np.array([])
    # for i in range(0, len(busy_car_places_id)):
    #     for j in range(0, len(bad_car_places_id)):
    #         if busy_car_places_id[i] != bad_car_places_id[j]:
    #             normal_car_places_id = np.append(normal_car_places_id, bad_car_places_id[j])

    # normal_car_places_id = np.unique(normal_car_places_id)
    # normal_car_places_id = normal_car_places_id.astype(np.int32)

    # for normal_place_id in normal_car_places_id:
    #     cv2.fillPoly(frame, [np.array(dict_parked_car_boxes[normal_place_id])], (235, 51, 35))
    """ Здесь все заебись, он выделяет только правильные места"""

    # for i in range(0, len(bad_car_places_id)):
    #     for j in range(0, len(normal_car_places_id)):
    #         if bad_car_places_id[i] != normal_car_places_id[j]:
    #             destroy_car_places_id = np.append(normal_car_places_id, bad_car_places_id[j])
    #
    # for destroy_place_id in destroy_car_places_id:
    #     cv2.fillPoly(frame, [np.array(dict_parked_car_boxes[destroy_place_id])], (50, 51, 35))

    cv2.addWeighted(frame, alpha, frame, 1 - alpha, 0, frame)

    # r = results[0]

    # frame = display_instances(frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    frame = display_instances(frame, results[0]['rois'], results[0]['masks'], results[0]['class_ids'], class_names,
                              results[0]['scores'])
    # classes = r['class_ids']
    classes = results[0]['class_ids']
    count_truck = 0
    count_car = 0
    # count_person = 0
    print("Total: ", len(classes))
    for i in range(len(classes)):
        if class_names[classes[i]] == 'car':
            count_car += 1
        # elif class_names[classes[i]] == 'truck':
        #     count_truck += 1
        # elif class_names[classes[i]] == 'person':
        #     count_person += 1

    return np.array(frame), count_truck, count_car, bad_cars_places_count, free_space


def count_to_pic(frame, count_truck, count_car, bad_cars_places_count, free_space):
    frame = cv2.putText(frame, f'auto: {count_car}', (300, 70), font, 1, (0, 0, 0), 1, cv2.LINE_4)
    # frame = cv2.putText(frame, f'truck: {count_truck}', (500, 70), font, 1, (0, 0, 0), 1, cv2.LINE_4)
    frame = cv2.putText(frame, f'bad places: {bad_cars_places_count}', (500, 70), font, 1, (0, 0, 0), 1, cv2.LINE_4)
    frame = cv2.putText(frame, f'free: {free_space}', (950, 70), font, 1, (0, 0, 0), 1, cv2.LINE_4)
    return np.array(frame)


class Config(mrcnn.config.Config):
    NAME = "model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 2
    #     NUM_CLASSES = 81
    MAX_GT_INSTANCES = 45
    # Сколько объектов на изображении детектятся
    DETECTION_MAX_INSTANCES = 45
    DETECTION_NMS_THRESHOLD = 0.35
    RPN_TRAIN_ANCHORS_PER_IMAGE = 128


class_names = ['BG', 'car']
# class_names = ['BG', 'person', 'bicycle', 'car', 'car', 'airplane',
#                'bus', 'train', 'truck', 'car', 'traffic light',
#                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
#                'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
#                'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
#                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#                'kite', 'baseball bat', 'baseball glove', 'car',
#                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
#                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
#                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
#                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
#                'keyboard', 'car', 'microwave', 'oven', 'toaster',
#                'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
#                'teddy bear', 'hair drier', 'toothbrush']

colors = random_colors(len(class_names))
class_dict = {name: color for name, color in zip(class_names, colors)}

config = Config()
config.display()

ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_car_atop1.h5")
# COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

print(COCO_MODEL_PATH)
if not os.path.exists(COCO_MODEL_PATH):
    mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)

model = MaskRCNN(mode="inference",
                 model_dir=MODEL_DIR, config=Config())
model.load_weights(COCO_MODEL_PATH, by_name=True)
regions = "regions.p"
with open(regions, 'rb') as f:
    parked_car_boxes = pickle.load(f)
    dict_parked_car_boxes = dict()
    for number in range(0, len(parked_car_boxes)):
        dict_parked_car_boxes.update({number: parked_car_boxes[number]})

dict_neighbors = dict()
for i in range(0, len(dict_parked_car_boxes.keys())):
    neighbors_list = []
    pol1_xy = Polygon(dict_parked_car_boxes[i])
    for j in range(0, len(dict_parked_car_boxes.keys())):
        pol2_xy = Polygon(dict_parked_car_boxes[j])
        polygon_intersection = pol1_xy.intersection(pol2_xy).area
        polygon_union = pol1_xy.union(pol2_xy).area
        IOU = polygon_intersection/polygon_union
        if IOU != 1:
            pol1_xy_bigger = shapely.affinity.scale(pol1_xy, xfact=2.0, yfact=2.0, origin='center')
            polygon_intersection = pol1_xy_bigger.intersection(pol2_xy).area
            polygon_union = pol1_xy_bigger.union(pol2_xy).area
            IOU = polygon_intersection/polygon_union
            if IOU != 0:
                neighbors_list.append(j)
    dict_neighbors.update({i: neighbors_list})

# parking_id_array = np.array([])
# for i in dict_parked_car_boxes.keys():
#     parking_id_array = np.append(parking_id_array, i)
# parking_id_array = parking_id_array.astype(np.int32)

alpha = 0.5
font = cv2.FONT_HERSHEY_SIMPLEX

app = Flask(__name__)


def gen(feed):
    while True:
        frame = feed.get_frame();
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(ParkingDetector()), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0')