import sys
import collections
import os
import time

import cv2
import numpy as np
from IPython import display
from numpy.lib.stride_tricks import as_strided
from openvino import inference_engine as ie
import ngraph as ng

from decoder_pose import OpenPoseDecoder
import player as utils

device_dict = {0:'CPU', 1:'GPU', 2: 'MULTI', 3:'HETERO'}
cur_device = 0


def pool2d(A, kernel_size, stride, padding, pool_mode="max"):
    A = np.pad(A, padding, mode="constant")

    output_shape = (
        (A.shape[0] - kernel_size) // stride + 1,
        (A.shape[1] - kernel_size) // stride + 1,
    )
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(
        A,
        shape=output_shape + kernel_size,
        strides=(stride * A.strides[0], stride * A.strides[1]) + A.strides
    )
    A_w = A_w.reshape(-1, *kernel_size)

    if pool_mode == "max":
        return A_w.max(axis=(1, 2)).reshape(output_shape)
    elif pool_mode == "avg":
        return A_w.mean(axis=(1, 2)).reshape(output_shape)


def heatmap_nms(heatmaps, pooled_heatmaps):
    return heatmaps * (heatmaps == pooled_heatmaps)


def process_results(img, results):
    pafs = results[output_keys[0]]
    heatmaps = results[output_keys[1]]

    pooled_heatmaps = np.array(
        [[pool2d(h, kernel_size=3, stride=1, padding=1, pool_mode="max") for h in heatmaps[0]]]
    )
    nms_heatmaps = heatmap_nms(heatmaps, pooled_heatmaps)

    poses, scores = decoder(heatmaps, nms_heatmaps, pafs)
    output_shape = exec_net.outputs[output_keys[0]].shape
    output_scale = img.shape[1] / output_shape[3], img.shape[0] / output_shape[2]
    poses[:, :, :2] *= output_scale

    return poses, scores


colors = ((255, 0, 255), (255, 0, 255), (255, 0, 255), (255, 0, 255), (255, 0, 255), (255, 0, 255),
          (255, 0, 255), (255, 0, 255), (255, 0, 255), (255, 0, 255), (255, 0, 255), (255, 0, 255),
          (255, 0, 255), (255, 0, 255), (255, 0, 255), (255, 0, 255), (255, 0, 255))

default_skeleton = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7),
                    (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6))


def draw_poses(img, poses, point_score_threshold, skeleton=default_skeleton):
    if poses.size == 0:
        return img

    img_limbs = np.copy(img)
    for pose in poses:
        points = pose[:, :2].astype(np.int32)
        points_scores = pose[:, 2]
        for i, (p, v) in enumerate(zip(points, points_scores)):
            if v > point_score_threshold:
                cv2.circle(img, tuple(p), 1, colors[i], 2)
        for i, j in skeleton:
            if points_scores[i] > point_score_threshold and points_scores[j] > point_score_threshold:
                cv2.line(img_limbs, tuple(points[i]), tuple(points[j]), color=colors[j], thickness=4)
    cv2.addWeighted(img, 0.4, img_limbs, 0.6, 0, dst=img)
    return img


def change_device(frame, f_width):
    global cur_device
    cur_device = (cur_device + 1) % len(device_dict)
    print(cur_device)
    cv2.putText(frame, f'{device_dict[cur_device]}', (550, 40),
                        cv2.FONT_HERSHEY_COMPLEX, f_width / 1000, (0, 0, 0), 1, cv2.LINE_AA)


def turn_img_into_video(img):
    images = []
    for i in range(1):
        images.append(img)
    height, width, layers = images[1].shape
    video = cv2.VideoWriter('custom.mp4', -1, 1, (width, height))
    for i in range(1):
        video.write(img)
    cv2.destroyAllWindows()
    return video


def hetero_usage(ie_core, net):
    # Нахождение оптимального количества запросов
    exec_net = ie_core.load_network(net, device_name="HETERO:GPU,CPU")
    nq = exec_net.get_metric("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
    print(nq)

    exec_net = ie_core.load_network(net, device_name="HETERO:CPU,GPU", num_requests=2)
    # Карта связывания для дефолтной конфигурации
    layers_map = ie_core.query_network(network=net, device_name="HETERO:CPU,GPU")
    cur_affinity = open(f'{path}\metrics\affinity.txt', 'w')
    for layer in layers_map:
        cur_affinity.write(f'{layer}: {layers_map[layer]}\n')    
    cur_affinity.close()

    # Новое связывание
    ng_func = ng.function_from_cnn(net)
    # Нахождение всех conv слоев
    conv_layers = []
    for layer in layers_map:
        index = layer.find('conv')
        if index != -1:
            conv_layers.append(layer)
    # Каждому conv -> GPU
    for i in conv_layers:
        layers_map[i] = 'GPU'
    # Присваивание
    for node in ng_func.get_ordered_ops():
        affinity = layers_map[node.get_friendly_name()]
        node.get_rt_info()["affinity"] = affinity
    # Новое связывание
    ie_core.set_config(config={'HETERO_DUMP_GRAPH_DOT':'YES'}, device_name='HETERO')
    exec_net = ie_core.load_network(net, device_name="HETERO:CPU,GPU")
    return exec_net


def run_pose_estimation(exec_net, source=0, flip=False, use_popup=False, skip_first_frames=0):
    player = None
    try:
        # инициализцаия видео плеера для воспроизведения с заданным fps
        player = utils.VideoPlayer(source, flip=flip, fps=30, skip_first_frames=skip_first_frames)
        player.start()
        if use_popup:
            title = "Press ESC to Exit"
            cv2.namedWindow(title, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)

        processing_times = collections.deque()
        while True:
            # захват кадра
            global cur_device
            frame = player.next()
            if frame is None:
                print("Source ended")
                break
            # преобразования, если разрешение больше FUllHD
            scale = 1280 / max(frame.shape)
            if scale < 1:
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            input_img = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            input_img = input_img.transpose(2, 0, 1)[np.newaxis, ...]

            # время инференса
            start_time = time.time()
            # результат
            results = exec_net.infer(inputs={input_key: input_img})
            stop_time = time.time()
            # получение поз из результатов сети
            poses, scores = process_results(frame, results)
            # отрисовка поз
            frame = draw_poses(frame, poses, 0.1)

            processing_times.append(stop_time - start_time)
            if len(processing_times) > 50:
                processing_times.popleft()

            _, f_width = frame.shape[:2]
            
            # средняя время инференса
            processing_time = np.mean(processing_times) * 1000
            metrics_latency[device_dict[cur_device]].append(processing_time)
            
            # пропускная способность
            fps = 1000 / processing_time
            metrics_FPS[device_dict[cur_device]].append(fps)
            
            # вывод статистики
            cv2.putText(frame, f"Время инференса: {processing_time:.1f} мс", (660, 40),
                        cv2.FONT_HERSHEY_COMPLEX, f_width / 1000, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"FPS: {fps:.1f}", (973, 90),
                        cv2.FONT_HERSHEY_COMPLEX, f_width / 1000, (0, 0, 255), 2, cv2.LINE_AA)
            
            # вывод текущего устройства исполнения     
            
            # cv2.rectangle(frame, (900, 690), (1280, 620), (255,255,255), cv2.FILLED) 
            cv2.putText(frame, f'Устройства: {device_dict[cur_device]}', (20, 40),
                            cv2.FONT_HERSHEY_COMPLEX, f_width / 1000, (255, 0, 0), 1, cv2.LINE_AA)


            if use_popup:
                cv2.imshow(title, frame)
                key = cv2.waitKey(1)
                # ESC
                if key == 27:                  
                    break
                # TAB
                if key == 9:    
                    cur_device = (cur_device + 1) % len(device_dict)
                    if cur_device == 0:
                        exec_net = ie_core.load_network(net, "CPU")
                    if cur_device == 1:
                        exec_net = ie_core.load_network(net, "GPU")
                    if cur_device == 2:
                        exec_net = ie_core.load_network(net, device_name="MULTI:GPU(4),CPU(2)")      
                    if cur_device == 3:
                        exec_net = ie_core.load_network(net, device_name="HETERO:GPU,CPU", num_requests=2)           
                        
            else:
                _, encoded_img = cv2.imencode(".jpg", frame, params=[cv2.IMWRITE_JPEG_QUALITY, 90])
                i = display.Image(data=encoded_img)
                display.clear_output(wait=True)
                display.display(i)
    except KeyboardInterrupt:
        print("Interrupted")
    except RuntimeError as e:
        print(e)
    finally:
        if player is not None:
            metrics.write('    MEAN                  MIN              MAX\n')
            metrics.write('Latency:\n')
            for i in metrics_latency:
                if not metrics_latency[i]:
                    metrics.write(f'  {i[:3]}:    None     \n')
                else:
                    metrics.write(f'  {i[:3]}: {str(np.round(np.median(np.array(metrics_latency[i])), 3))} ms     ')
                    metrics.write(f'    {str(np.round(np.min(np.array(metrics_latency[i])), 3))} ms \n')
            metrics.write('FPS:\n')
            for i in metrics_FPS:
                if not metrics_FPS[i]:
                    metrics.write(f'  {i[:3]}:    None     \n')
                else:
                    metrics.write(f'  {i[:3]}: {str(np.round(np.median(np.array(metrics_FPS[i])), 3))}            ')
                    metrics.write(f'                 {str(np.round(np.max(np.array(metrics_FPS[i])), 3))}\n')
            player.stop()
        if use_popup:
            metrics.close()
            cv2.destroyAllWindows()

path = os.path.abspath(os.path.dirname(__file__))

metrics_latency = {'CPU': [], 'GPU': [], 'MULTI': [], 'HETERO': []}
metrics_FPS = {'CPU': [], 'GPU': [], 'MULTI': [], 'HETERO': []}
metrics = open(f'{path}\metrics\metrics.txt', 'w')

# директория, куда загружены модели
base_model_dir = f'{path}\model'
# название модели
model_name = 'human-pose-estimation-0001'
# выбранная точность (FP32, FP16, FP16-INT8)
precision = 'FP16'

model_path = f"{base_model_dir}\{precision}\{model_name}.xml"
model_weights_path = f"{base_model_dir}\{precision}\{model_name}.bin"


# Инициализация inference engine
ie_core = ie.IECore()

# Загрузка модели в виде IR
net = ie_core.read_network(model=model_path, weights=model_weights_path)

# Загрузка модели для выполнения на целевом устройстве
exec_net = ie_core.load_network(net, device_name="CPU")

# exec_net = ie_core.load_network(net, device_name="HETERO:CPU,GPU", num_requests=2)

# layers_map = ie_core.query_network(network=net, device_name="HETERO:CPU,GPU")
# cur_affinity = open(f'{path}\\metrics\\affinity.txt', 'w')
# for layer in layers_map:
#     cur_affinity.write(f'{layer}: {layers_map[layer]}\n')     
# cur_affinity.close()

# ng_func = ng.function_from_cnn(net)

# conv_layers = []
# for layer in layers_map:
#     index = layer.find('conv')
#     if index != -1:
#         conv_layers.append(layer)

# for i in conv_layers:
#     layers_map[i] = 'GPU'


# for node in ng_func.get_ordered_ops():
#     affinity = layers_map[node.get_friendly_name()]
#     node.get_rt_info()["affinity"] = affinity

# ie_core.set_config(config={'HETERO_DUMP_GRAPH_DOT':'YES'}, device_name='HETERO')
# exec_net = ie_core.load_network(net, device_name="HETERO:CPU,GPU")



# Названия входного и выходного слоев
input_key = list(exec_net.input_info)[0]
output_keys = list(exec_net.outputs.keys())

# Параметры входного слоя
height, width = exec_net.input_info[input_key].tensor_desc.dims[2:]

# Декодер для поз
decoder = OpenPoseDecoder()            
            
# Файл для обработки        
video_file = 'https://github.com/GandhiKK/OpenVINO_hetero/blob/master/smert/diploma/data/gym-detection4x.mp4?raw=true'

# Обработка видео файла
run_pose_estimation(exec_net, video_file, flip=False, use_popup=True, skip_first_frames=500)

# Обработка видео с веб-камеры
# run_pose_estimation(source=0, flip=True, use_popup=True)