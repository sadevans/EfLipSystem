# Веб-приложение для визуального распознавания произнесенных слов на видео

Данный репозиторий содержит в себе разработанное веб-приложение для взаимодействия с обученной моделью. Архитектура модели подгружается в качестве сабмодуля. 

Серверная часть написана на Django, клиентская - на Vanilla JS.

## Установка зависимостей
Для того, чтобы установить необходимую среду разработки, необходимо:

### 1. 

### 2. 

## Описание конфигурации проекта

- Файл `setup.py` отвечает за сборку бэкенд части проекта.
- В папке `model` находится сабмодуль - проект с моделью.
- Модуль `pipelines` содержит в себе пайплайн, который является связующим звеном между бэкенд частью проекта и моделью. В модуле `pipelines` содержится файл `pipelines.py`, который реализует логику обработки данных для подачи в модель, вызов модели, вызов обработки результирующего видео.
- Модуль `pipelines/utils` содержит в себе скрипт обработки результирующего видео - наложение области интереса и субтитров на исходное видео.
- Модуль `pipelines/data` содержит в себе скрипты для обработки входного видеофайла для его подачи в модель.
- Модуль `pipelines/detectors` содержит в себе скрипты для обнаружения лиц на кадре и поиска ключевых точек с помощью детектора **mediapipe**.
- Модуль `backend/lipread/templates` содержит в себе html шаблон веб-страницы.
- Модуль `backend/lipread/static` содержит в себе *.css* и *.js* скрипты.
- Модуль `backend/lipread/lipread` содержит в себе базовые файлы фреймворка Django. 
- Модуль `backend/lipread/lipread/core` содержит в себе основную бэкенд логику проекта. В корне этого модуля содержатся стандартные скрипты, генерируемые Django.

## Принцип работы

**Принцип работы программного комплекса инференса модели представлен ниже**

![program_infer](https://github.com/sadevans/EfLipSystem/assets/82286355/4fede64a-3d2b-4d0f-b3f3-55e6857c1b78)

Схема работы состоит из трех этапов:
1. Кадры исходного видео поступают из веб-приложения от клиента и прередобрабатываются. Итогом выполнения этого шага является область интереса на кадрах видео.
2. Область интереса обрабатывается модель. Результатом этого шага является текст. произнесенный на видео.
3. Затем текст и область интереса накладываются на исходное видео и отдаются обратно в клиентскую часть веб-приложения.

Рассмотрим каждый шаг подробнее.

### Предобработка кадров видео
На этом этапе с помощью `mediapipe` находится лицо человека на кадре, координаты его bounding box'а сохраняются для последующего использования. После этого ищутся ключевые точки лица, их в этом случае 4 - правый глаз, левый глаз, кончик носа и середина рта. Далее с помощью найденных ключевых точек на кадрах видео обрезается область интереса размером 88х88 пикселей.


**Диаграмма последовательности работы веб-приложения**

![uml_web_app (1)](https://github.com/sadevans/EfLipSystem/assets/82286355/0554a098-7a8d-43e1-9862-613adc4374a5)

