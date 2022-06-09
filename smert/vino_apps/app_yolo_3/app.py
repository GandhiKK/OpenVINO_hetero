# !/usr/bin/env python
from __future__ import print_function, division

import logging
import sys
from math import exp as exp
from time import time
import ngraph as ng

import cv2
from openvino.inference_engine import IECore

logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO, stream=sys.stdout)


class Parameters:
    def __init__(self, param, side):
        self.num = 3 if 'num' not in param else int(param['num'])
        self.coords = 4 if 'coords' not in param else int(param['coords'])
        self.classes = 80 if 'classes' not in param else int(param['classes'])
        self.side = side
        self.anchors = param['anchors']

        if param.get('mask'):
            mask = param['mask']
            self.num = len(mask)

            maskedAnchors = []
            for idx in mask:
                maskedAnchors += [self.anchors[idx * 2], self.anchors[idx * 2 + 1]]
            self.anchors = maskedAnchors


def getIndex(side, coord, classes, location, entry):
    return int(side ** 2 * (location // side ** 2 * (coord + classes + 1) + entry) + location % side ** 2)


def scale_bbox(x, y, h, w, class_id, confidence, h_scale, w_scale):
    return dict(
        xmin=int((x - w / 2) * w_scale),
        xmax=int(int((x - w / 2) * w_scale) + w * w_scale),
        ymin=int((y - h / 2) * h_scale),
        ymax=int(int((y - h / 2) * h_scale) + h * h_scale),
        class_id=class_id,
        confidence=confidence
    )


def parse_region(blob, resized_shape, original_shape, params, threshold):
    original_height, original_width = original_shape
    resized_height, resized_width = resized_shape
    objects = list()
    predictions = blob.flatten()
    square = params.side * params.side

    for i in range(square):
        row = i // params.side
        col = i % params.side
        for n in range(params.num):
            predict_index = getIndex(params.side, params.coords, params.classes, n * square + i, params.coords)
            predict = predictions[predict_index]
            if predict < threshold:
                continue

            predict_index = getIndex(params.side, params.coords, params.classes, n * square + i, 0)

            x_index = (col + predictions[predict_index + 0 * square]) / params.side
            y_index = (row + predictions[predict_index + 1 * square]) / params.side

            try:
                width_exp = exp(predictions[predict_index + 2 * square])
                height_exp = exp(predictions[predict_index + 3 * square])
            except OverflowError:
                continue

            width = width_exp * params.anchors[2 * n] / resized_width
            height = height_exp * params.anchors[2 * n + 1] / resized_height
            for class_id in range(params.classes):
                predict_index = getIndex(params.side, params.coords, params.classes, n * square + i,
                                         params.coords + 1 + class_id)
                predict = predict * predictions[predict_index]
                if predict < threshold:
                    continue
                objects.append(
                    scale_bbox(x=x_index, y=y_index, h=height, w=width, class_id=class_id, confidence=predict,
                               h_scale=original_height, w_scale=original_width))
    return objects


def intersection(box_1, box_2):
    width = min(box_1['xmax'], box_2['xmax']) - max(box_1['xmin'], box_2['xmin'])
    height = min(box_1['ymax'], box_2['ymax']) - max(box_1['ymin'], box_2['ymin'])
    if width < 0 or height < 0:
        area = 0
    else:
        area = width * height
    area_1 = (box_1['ymax'] - box_1['ymin']) * (box_1['xmax'] - box_1['xmin'])
    area_2 = (box_2['ymax'] - box_2['ymin']) * (box_2['xmax'] - box_2['xmin'])
    area_union = area_1 + area_2 - area
    if area_union == 0:
        return 0
    return area / area_union


def main():
    
    ie = IECore() # Инициализация ядра
    
    # Входные параметры

    model_name = 'D:\\Extra\\pyDev\\smert\\openvino\\app_yolo_3\\frozen_darknet_yolov3_model' # навзание модели
    # Файл с классами объектов
    label_file = 'D:\\Extra\\pyDev\\smert\\openvino\\app_yolo_3\\coco.names.txt'
    # Тестовое видео
    test_file = 'D:\\Extra\\pyDev\\smert\\openvino\\app_yolo_3\\test.mp4'
    # Минимальная увереность модели в результате
    prob_threshold = 0.5
    # Минимальная степень пересечения между прямоугольниками
    iou_threshold = 0.4

    
    

    # Загрузка IR-модели
    net = ie.read_network(model_name + '.xml', model_name + ".bin")

    # Размер batch
    net.batch_size = 1

    # Параметры принимаемого изображения (входного слоя)
    n, c, h, w = net.input_info['inputs'].input_data.shape

    # Загрузка маркировки классов
    with open(label_file, 'r') as f:
        labels_map = [x.strip() for x in f]

    async_mode = True
    wait_key_code = 1

    # Загрузка видео или изображения
    cap = cv2.VideoCapture(test_file)

    # В случае если загружается только одно изображение или директория с изображениями,
    # то приложение проходит по всем изображениям синхроно
    number_input_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    number_input_frames = 1 if number_input_frames != -1 and number_input_frames < 0 else number_input_frames
    if number_input_frames != 1:
        ret, frame = cap.read()
    else:
        async_mode = False
        wait_key_code = 0

    # Загрузка нейронной сети для выполнения на целевом устройстве
    exec_net = ie.load_network(network=net, num_requests=2, device_name='CPU')

    # Вспомогательные примитивы
    cur_request_id = 0
    next_request_id = 1
    render_time = 0
    parsing_time = 0
    function = ng.function_from_cnn(net)

    while cap.isOpened():
        # Получение изображения
        if async_mode:
            ret, next_frame = cap.read()
        else:
            ret, frame = cap.read()

        if not ret:
            break

        if async_mode:
            request_id = next_request_id
            in_frame = cv2.resize(next_frame, (w, h))
        else:
            request_id = cur_request_id
            in_frame = cv2.resize(frame, (w, h))

        # Изменение размерности изображения
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        in_frame = in_frame.reshape((n, c, h, w))

        # Запуск нейронной сети на ЦП
        start_time = time()
        exec_net.start_async(request_id=request_id, inputs={'inputs': in_frame})
        det_time = time() - start_time

        # Сбор результатов детектирования
        objects = list()
        if exec_net.requests[cur_request_id].wait(-1) == 0:
            start_time = time()
            output = exec_net.requests[cur_request_id].outputs
            for layer_name, out_blob in output.items():
                out_blob = out_blob.reshape(net.outputs[layer_name].shape)
                params = [x._get_attributes() for x in function.get_ordered_ops() if x.get_friendly_name() == layer_name][0]
                layer_params = Parameters(params, out_blob.shape[2])
                objects += parse_region(out_blob, in_frame.shape[2:], frame.shape[:-1], layer_params, prob_threshold)
            parsing_time = time() - start_time

        objects = sorted(objects, key=lambda obj: obj['confidence'], reverse=True)
        for i in range(len(objects)):
            if objects[i]['confidence'] == 0:
                continue
            for j in range(i + 1, len(objects)):
                if intersection(objects[i], objects[j]) > iou_threshold:
                    objects[j]['confidence'] = 0

        # Фильтрация объектов, в конечный пул попадают только объекты с уверенностью правильности свыше 0.5
        objects = [obj for obj in objects if obj['confidence'] >= prob_threshold]

        origin_im_size = frame.shape[:-1]
        for obj in objects:
            # Проверка на обноружение объекта
            if obj['xmax'] > origin_im_size[1] or obj['ymax'] > origin_im_size[0] or obj['xmin'] < 0 or obj['ymin'] < 0:
                continue

            # Прямоугольника вокруг объекта
            color = (
                int(min(obj['class_id'] * 12.5, 255)), min(obj['class_id'] * 7, 255), min(obj['class_id'] * 5, 255))
            cv2.rectangle(frame, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), color, 2)

            # Надпись рядом с обхектом
            det_label = labels_map[obj['class_id']] if labels_map and len(labels_map) >= obj['class_id'] else str(
                obj['class_id'])
            text = "#" + det_label + ' ' + str(round(obj['confidence'] * 100, 1)) + ' %'
            coordinate = (obj['xmin'], obj['ymin'] - 7)
            cv2.putText(frame, text, coordinate, cv2.FONT_HERSHEY_COMPLEX, 1, color, 1)

        # Отрисовка метрик
        inf_time_message = "Время предсказания: не для асинхронного режима" if async_mode else \
            "Время предсказания: {:.3f} мс".format(det_time * 1e3)
        render_time_message = "Отрисовка выделения объектов: {:.3f} мс".format(render_time * 1e3)
        parsing_message = "Маркировка {:.3f} мс".format(parsing_time * 1e3)
        cv2.putText(frame, inf_time_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
        cv2.putText(frame, parsing_message, (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
        cv2.putText(frame, render_time_message, (15, 45), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)

        # Показ кадра
        start_time = time()
        cv2.imshow("img", frame)
        render_time = time() - start_time

        if async_mode:
            cur_request_id, next_request_id = next_request_id, cur_request_id
            frame = next_frame

        key = cv2.waitKey(wait_key_code)

        # ESC key
        if key == 27:
            break
        # Tab key
        if key == 9:
            exec_net.requests[cur_request_id].wait()
            async_mode = not async_mode

    cv2.destroyAllWindows()


if __name__ == '__main__':
    sys.exit(main() or 0)
