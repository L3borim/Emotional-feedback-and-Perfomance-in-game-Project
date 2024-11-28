import cv2
import numpy as np
import dlib
import pandas as pd
import time
from cv2 import dnn
from math import ceil

# Configurações do modelo ONNX
image_mean = np.array([127, 127, 127])
image_std = 128.0
iou_threshold = 0.3
center_variance = 0.1
size_variance = 0.2
min_boxes = [[10.0, 16.0, 24.0], [32.0, 48.0], [64.0, 96.0], [128.0, 192.0, 256.0]]
strides = [8.0, 16.0, 32.0, 64.0]
threshold = 0.5

# Dicionário de emoções
emotion_dict = {
    0: 'neutral',
    1: 'happiness',
    2: 'surprise',
    3: 'sadness',
    4: 'anger',
    5: 'disgust',
    6: 'fear'
}

# Funções auxiliares para pré-processamento
def define_img_size(image_size):
    shrinkage_list = []
    feature_map_w_h_list = []
    for size in image_size:
        feature_map = [int(ceil(size / stride)) for stride in strides]
        feature_map_w_h_list.append(feature_map)

    for i in range(0, len(image_size)):
        shrinkage_list.append(strides)
    priors = generate_priors(
        feature_map_w_h_list, shrinkage_list, image_size, min_boxes
    )
    return priors


def generate_priors(feature_map_list, shrinkage_list, image_size, min_boxes):
    priors = []
    for index in range(0, len(feature_map_list[0])):
        scale_w = image_size[0] / shrinkage_list[0][index]
        scale_h = image_size[1] / shrinkage_list[1][index]
        for j in range(0, feature_map_list[1][index]):
            for i in range(0, feature_map_list[0][index]):
                x_center = (i + 0.5) / scale_w
                y_center = (j + 0.5) / scale_h

                for min_box in min_boxes[index]:
                    w = min_box / image_size[0]
                    h = min_box / image_size[1]
                    priors.append([x_center, y_center, w, h])
    return np.clip(priors, 0.0, 1.0)


def convert_locations_to_boxes(locations, priors, center_variance, size_variance):
    if len(priors.shape) + 1 == len(locations.shape):
        priors = np.expand_dims(priors, 0)
    return np.concatenate([
        locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
        np.exp(locations[..., 2:] * size_variance) * priors[..., 2:]
    ], axis=len(locations.shape) - 1)


def center_form_to_corner_form(locations):
    return np.concatenate(
        [locations[..., :2] - locations[..., 2:] / 2,
         locations[..., :2] + locations[..., 2:] / 2],
        len(locations.shape) - 1
    )


# Inicializar o detector de rostos (Dlib) e carregar o modelo ONNX
detector_dlib = dlib.get_frontal_face_detector()
model = cv2.dnn.readNetFromONNX('RFB-320/emotion-ferplus-8.onnx')

input_size = [320, 240]
width, height = input_size
priors = define_img_size(input_size)

# Variáveis importantes
video = cv2.VideoCapture('./videos/video_test1.mp4')
emotion_data = []
start_time = time.time()
frame_count = 0

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (640, 360))
    elapsed_time = time.time() - start_time

    # Converter para escala de cinza
    gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    # Detectar rostos
    faces = detector_dlib(gray_frame)

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        # Garantir que as coordenadas estejam dentro dos limites do frame
        x = max(0, x)
        y = max(0, y)
        w = min(w, gray_frame.shape[1] - x)
        h = min(h, gray_frame.shape[0] - y)

        if w < 20 or h < 20:
            continue  # Ignorar rostos muito pequenos

        # Recortar a região do rosto
        face_frame = gray_frame[y:y+h, x:x+w]

        # Verificar se o recorte não está vazio
        if face_frame.size == 0:
            continue

        try:
            # Redimensionar e ajustar para o modelo
            resized_face = cv2.resize(face_frame, (64, 64)).reshape(1, 1, 64, 64)
            model.setInput(resized_face)

            # Fazer predição
            output = model.forward()
            emotion_index = np.argmax(output)
            emotion = emotion_dict.get(emotion_index, 'unknown')

            # Registrar dados e desenhar no quadro
            emotion_data.append({'Time': elapsed_time, 'Emotion': emotion})
            cv2.putText(frame_resized, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (255, 0, 0), 2)
        except Exception as e:
            print(f"Erro ao processar o rosto: {e}")

    cv2.imshow('Emotion Recognition', frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()


# Salvar os dados em um arquivo CSV
df = pd.DataFrame(emotion_data)
df.to_csv('emotion_detection_results.csv', index=False)
print("Resultados de emoções salvos em 'emotion_detection_results.csv'")
