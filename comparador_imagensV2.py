import cv2
import face_recognition as fr
import os

#encontra o caminho para a imagem
base_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(base_dir, "imgs")

#inicializa o cv2 na captura da webcam, diminui o fps e codifica a imagem
webcam = cv2.VideoCapture(0)

imgDilma = fr.load_image_file(image_dir + '/dilma_rousseff_image.jpg')
encodeImgDilma = fr.face_encodings(imgDilma)[0]

process_frame = True
while True:

    verificador, frame = webcam.read()
    
    if process_frame:

        if not verificador:
            print("não foi possível se conectar com a webcam")
            break
            
        # Redimensione o quadro do vídeo para 1/4 do tamanho para um processamento de reconhecimento facial mais rápido
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Converta a imagem da cor BGR (que o OpenCV usa) para a cor RGB (que usa o face_recognition)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        facelocations = fr.face_locations(rgb_small_frame)
        encodeImgFrames = fr.face_encodings(rgb_small_frame, facelocations)

        known_match_faces = []

        for encodeImgFrame in encodeImgFrames:
            known_match_faces.append(fr.compare_faces([encodeImgDilma], encodeImgFrame))
        
    process_frame = not process_frame

    for (top, right, bottom, left), match in zip(facelocations, known_match_faces):
        # Redimensione os locais do rosto, pois o quadro em que detectamos foi dimensionado para 1/4 do tamanho
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            if match == [True]: 
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    cv2.imshow("reconhecimento facial", frame)

    if cv2.waitKey(1) == 27:
        break

webcam.release()
cv2.destroyAllWindows()