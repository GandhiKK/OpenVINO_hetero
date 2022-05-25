# Pose Estimation Hetero

* [Текст](#текст)
* [Требования](#требования)
* [Актуальность и тд](#актуальность)
* [Видео пример](#пример-использования)

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

Готовый текст [тут](Документы/Основа/Диплом.pdf)

* [x] Глава 1
* [x] Глава 2
* [x] Глава 3
* [x] Глава 4
* [x] Глава 5

---

### Актуальность

Компьютерное зрение – одна из тех технологий, которые в последние несколько лет развиваются скачками. Если оглянуться на 10 лет назад, то это было не так, поскольку данная область была скорее темой академического интереса. Однако сейчас компьютерное зрение явно является движущей силой и помощником искусственного интеллекта. Основной проблемой данного подхода является большое количество требуемых ресурсов, что вызывает некоторые сложности при использовании его на практике, такие как необходимость больших финансовых вложений. Они необходимы не только при использовании самого алгоритма, но и для его обучения и сбора данных для этого же обучения, для чего требуется еще и непомерное количество времени.

Решая эту проблему компания Intel создала программное обеспечение OpenVINO Toolkit, позволяющее оптимизировать уже созданные ранее модели и выполнять их на вычислительных устройствах компании Intel с гораздо большой эффективностью. Данный инструмент предоставляет возможность использования нескольких устройств для инференса нейронных сетей. Использование Intel OpenVINO Toolkit для распределенной обработки данных позволяет:

* Использовать мощность ускорителей для обработки наиболее тяжелых частей сети и выполнять неподдерживаемые уровни на резервных устройствах, таких как CPU.
* Эффективнее использовать все доступные аппаратные средства в течение одного инференса, что дает более стабильную производительность, так как устройства разделяют нагрузку на выводы
* Повышение пропускной способности за счет использования нескольких устройств (по сравнению с выполнением на одном устройстве).

### Проблема

Высокое потребеление ресурсов при применении глубокого обучения.

### Цель

Проанализировать эффективность оптимизации инференса инструмента OpenVINO Toolkit в случае распределенной обработки информации

### Объект исследования

Функционал OpenVINO Toolkit для распредленной обработки данных

### Предмет исследования

Процесс оптимизации инференса нейросетей с помощью плагинов для распределенной обработки

### Задачи

* Изучить фукнционал OpenVINO Toolkit для распредленной обработки данных
* Разработать систему для оценки оптимизации инференса
* Проанализировать эффективность оптимизации инференса данного инструмента в случае использования мультидевайсного плагина
* Проанализировать эффективность оптимизации инференса данного инструмента в случае использования гетерогенного плагина

---

## Пример использования

https://user-images.githubusercontent.com/42946613/168580222-dbfd962d-e160-40b3-8284-2b7dcb044351.mp4
