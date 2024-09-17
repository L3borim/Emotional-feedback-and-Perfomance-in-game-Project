import cv2
import dlib
from fer import FER  
import pandas as pd
import time

# Inicializar o detector de rosto dlib
detector_dlib = dlib.get_frontal_face_detector()

# Inicializar o detector de emoções FER
detector_fer = FER(mtcnn=True)

def analyze_emotions(frame):
    try:
        emotion, score = detector_fer.top_emotion(frame)
        if emotion and score > 0.5:
            return emotion
        else:
            return None
    except Exception as e:
        print(f"Erro na análise de emoções: {e}")  #log de erro para depuração
        return None

# Variáveis importantes
video = cv2.VideoCapture('./videos/video_overwatch02.mp4')
emotion_data = []
start_time = time.time()
frame_count = 0

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Redimensionamento do quadro para acelerar o processamento
    frame_resized = cv2.resize(frame, (640, 360))
    
    elapsed_time = time.time() - start_time
    
    # Processar quadro a quadro
    if frame_count % 1 == 0:
        # Converter o quadro para escala de cinza para detecção de rostos
        gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        
        # Detectar rostos usando dlib
        faces = detector_dlib(gray_frame)
        
        for face in faces:
            # Obter coordenadas do rosto detectado
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())

            # Verificar se o rosto é grande o suficiente para análise
            if w < 20 or h < 20:
                continue  # Ignora rostos muito pequenos que podem causar erros

            # Recortar a região do rosto do quadro
            face_frame = frame_resized[y:y+h, x:x+w]

            # Analisar emoções apenas na região do rosto
            emotion = analyze_emotions(face_frame)
            if emotion:
                # Adicionar emoção e tempo à lista
                emotion_data.append({'Time': elapsed_time, 'Emotion': emotion})
                
                # Desenhar o texto da emoção dominante no quadro original
                cv2.putText(frame_resized, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                # Desenhar retângulo ao redor do rosto
                cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    frame_count += 1

    cv2.imshow('Emotion Recognition', frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

df = pd.DataFrame(emotion_data)
df.to_csv('emotion_detection_results.csv', index=False)
print("Resultados de emoções salvos em 'emotion_detection_results.csv'")
