from keras.models import load_model
import cv2
import numpy as np
import cvzone
import os

class ImageClassifier:
    def __init__(self, model_path, class_names):
        self.model = load_model(model_path, compile=False)
        self.class_names = class_names

    def preprocess_image(self, image_path):
        img = cv2.imread(image_path)
        img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        image_array = np.asarray(img_resized, dtype=np.float32).reshape(1, 224, 224, 3)
        normalized_image = (image_array / 127.5) - 1
        return normalized_image, img

    def predict_image(self, image_path):
        normalized_image, original_image = self.preprocess_image(image_path)
        prediction = self.model.predict(normalized_image)
        index = np.argmax(prediction)
        class_name = self.class_names[index]
        confidence_score = prediction[0][index]
        return class_name, confidence_score, original_image

    def display_result(self, class_name, confidence_score, image):
        text1 = f"Tipo: {class_name}"
        text2 = f"Taxa de acertividade: {str(np.round(confidence_score * 100))[:-2]} %"
        img_resized = cv2.resize(image, (750, 750), interpolation=cv2.INTER_AREA)
        cvzone.putTextRect(img_resized, text1, (50, 50), scale=2)
        cvzone.putTextRect(img_resized, text2, (50, 100), scale=2)
        cv2.imshow('IMG', img_resized)
        cv2.waitKey(0)

def main():
    model_path = "keras_Model.h5"
    class_names = ['Parasita', 'normal']

    # Solicitar ao usuário para escolher entre "normal" e "parasita"
    folder_choice = input("Digite 'normal' ou 'parasita' para selecionar a pasta de imagens: ").lower()
    if folder_choice not in ['normal', 'parasita']:
        print("Escolha inválida. Digite 'normal' ou 'parasita'.")
        return

    folder_path = os.path.join("images", folder_choice)
    image_files = os.listdir(folder_path)

    if not image_files:
        print(f"A pasta '{folder_choice}' está vazia. Adicione imagens antes de executar o programa.")
        return

    # Solicitar ao usuário para escolher uma imagem da pasta selecionada
    print(f"Imagens disponíveis em '{folder_choice}':")
    for i, image in enumerate(image_files):
        print(f"{i + 1}. {image}")

    image_choice = input(f"Digite o número da imagem desejada (1-{len(image_files)}): ")
    try:
        image_index = int(image_choice) - 1
        if 0 <= image_index < len(image_files):
            image_path = os.path.join(folder_path, image_files[image_index])

            classifier = ImageClassifier(model_path, class_names)
            class_name, confidence_score, image = classifier.predict_image(image_path)
            classifier.display_result(class_name, confidence_score, image)

        else:
            print("Escolha inválida. Digite um número válido.")
    except ValueError:
        print("Entrada inválida. Digite um número.")

if __name__ == "__main__":
    main()
