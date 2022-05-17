# Pose Estimation Hetero

* [Требования](#требования)
* [Текст](#текст)
* [Актуальность и тд](#актуальность)
* [Видео пример использования](#пример-использования)

## Требования

* python 3.6 и выше
* pip install -r requirements.txt
* В файле:
  * Выбрать точность модели (**FP16** / FP32 / FP16-INT8)
  * Обработка видео (**файл** / веб-камера)
* python human_pose_est.py
* Функции
  * TAB - изменить устройство для инференса
  * ESC - заверешение работы

---

## Текст

Посмотреть текст [тут](Документы/Основа/Диплом.pdf)

* [x] Глава 1
* [x] Глава 2
* [ ] Глава 3
* [ ] Глава 4

---

### Актуальность

На данный момент самой популярной и эффективной технологией для компьютерного зрения является глубокое обучение. Именно этот подход и позволил обогнать человека в эффективности по распознаванию объектов на изображении. Проблема данного подхода в том, что он требует много ресурсов. По этой причине его сложно применять на практике, либо внедрение подхода может обойтись в большую сумму.

Это касается не только работы самого алгоритма, но и обязательных подготовительных мероприятий, таких как обучение, которое может занимать кучу времени и ресурсов. А если брать в учёт ещё и сбор обучающей выборки, то счёт может пойти на месяца. Решая эту проблему компания Intel создала программное обеспечение OpenVINO Toolkit, позволяющее оптимизировать уже созданные ранее модели и алгоритмы и выполнять их на вычислительных устройствах компании Intel с гораздо большой эффективностью в сравнении со стандартным использованием.

В частности использование OpenVINO для распредленной обработки позволяет:

* Использовать мощность ускорителей для обработки наиболее тяжелых частей сети и выполнять неподдерживаемые уровни на резервных устройствах, таких как CPU.

* Эффективнее использовать все доступные аппаратные средства в течение одного инференса, что дает более стабильную производительность, так как устройства разделяют нагрузку на выводы

* Повышение пропускной способности за счет использования нескольких устройств (по сравнению с выполнением на одном устройстве)

### Проблема

Высокое потребление ресурсов при применении глубокого обучения

### Цель

Исследовать функционал OpenVINO Toolkit и проанализировать эффективность оптимизации инференса данного инструмента в случае распредленной обработки информации

### Объект исследования

Функционал OpenVINO Toolkit для распредленной обработки данных

### Предмет исследования

Процесс оптимизации инференса нейросетей с помощью плагинов для распределенной обработки

### Задачи

* Изучить фукнционал OpenVINO Toolkit для распредленной обработки данных
* Разработать систему для оценки оптимизации инференса
* Проанализировать эффективность оптимизации инференса данного инструмента в случае использования гетерогенного плагина
* Проанализировать эффективность оптимизации инференса данного инструмента в случае использования мультидевайсного плагина

---

## Использование

### Пример метрик

[Метрики](smert/diploma/metrics/metrics_def.txt)

### Видео

https://user-images.githubusercontent.com/42946613/168580222-dbfd962d-e160-40b3-8284-2b7dcb044351.mp4

