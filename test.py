from PIL import Image
from lang_sam import LangSAM
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import torch
from lang_sam import utils
import tools
from paho.mqtt import client as mqtt_client
import random
import json

def get_args_parser():
    parser = argparse.ArgumentParser(description="LangSAM Inference")
    parser.add_argument("--image_path", type=str, help="Path to the input image")
    parser.add_argument("--video_path", type=str, default="./assets/Test.mp4", help="Path to the input video")
    parser.add_argument("--text_prompt", type=str, default="wheel, car, truck.", help="Text prompt for segmentation")
    parser.add_argument("--output_path", type=str, default="output", help="Directory to save the output")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame for image processing")
    parser.add_argument("--end_frame", type=int, default=None, help="End frame for image processing")
    parser.add_argument("--confidence_threshold", type=float, default=0.45, help="Confidence threshold for predictions")
    return parser

def combine_all_masks(image_np, masks):
    # Sicherstellen, dass alle Masken NumPy-Arrays sind
    masks = [np.array(mask) for mask in masks]

    # Erstelle eine leere Maske mit der gleichen Form wie das Eingangsbild
    combined_mask = np.zeros_like(image_np)

    # Iteriere über alle Masken und kombiniere sie
    for mask in masks:
        # Erweitere die Maske auf 3 Kanäle (RGB)
        mask_3d = np.stack([mask] * 3, axis=-1)

        # Wähle eine Farbe für die Maske (z. B. Rot)
        mask_color = [255, 0, 0]

        # Überlagere die Maske auf das kombinierte Bild
        combined_mask = np.where(mask_3d > 0, mask_color, combined_mask)

    # Kombiniere die Maske mit dem Eingangsbild
    overlay = np.where(combined_mask > 0, combined_mask, image_np)

    return overlay

def ImageInference(image_pil, text, start_frame, end_frame, model, confidence_threshold, kalman):
    if start_frame != 0 and end_frame is not None:
        if start_frame > end_frame:
            raise ValueError("start_frame must be less than or equal to end_frame")
        else:
            id = start_frame
            while id <= end_frame:
                image_pil = Image.open(f"{args.image_path}/frame_{id}.jpg").convert("RGB")
                image_np = np.array(image_pil)
                results = model.predict([image_pil], [text], [0.45])
                first_result = results[0]
                masks = first_result["masks"]
                probs = first_result["scores"]
                labels = first_result["labels"]
                xyxy = first_result["boxes"]

                if masks is None or len(masks) == 0:
                        overlay = image_np
                        print("No masks found for the given text prompt.")

                else:
                    #overlay = combine_all_masks(image_np, masks)
                    overlay = utils.draw_image(image_np, masks, xyxy, probs, labels)
                    output_path = f"{args.output_path}/output_frame{id}.jpeg"
                    plt.imsave(output_path, overlay.astype(np.uint8))
                    print(f"Processed image saved to {output_path}")
                id += 1



    else:
        image_np = np.array(image_pil)
        results = model.predict([image_pil], [text])


        # Alle Keys aus Ergebnissen extrahieren
        first_result = results[0]
        masks = first_result["masks"]
        scores = first_result["scores"]
        labels = first_result["labels"]
        xyxy = first_result["boxes"]


        if masks is None or len(masks) == 0:
            overlay = image_np
            print("No masks found for the given text prompt.")
            client.publish(topic_1, "No masks found for the given text prompt.")

        else:
            # Filtere Scores, Boxen und Labels basierend auf dem Schwellwert
            valid_indices = scores >= confidence_threshold  # Nur Scores >= 0.45 berücksichtigen
            scores = scores[valid_indices]
            xyxy = xyxy[valid_indices]
            labels = [labels[i] for i in range(len(labels)) if valid_indices[i]]
            masks = masks[valid_indices]

            message = []
            for bbox, label in zip(xyxy, labels):
                message.append([bbox, label])
                message_json = json.dumps(message)

                return message_json

            # Sende die Nachricht über MQTT
            client.publish(topic_1, message_json)

        if len(scores) == 0:
            overlay = image_np
            print("No valid detections after applying the threshold.")

        else:
            for index, label in enumerate(labels):
                if label == "car":
                    x_min = int(xyxy[index - 1][0])
                    y_min = int(xyxy[index - 1][1])
                    x_max = int(xyxy[index - 1][2])
                    y_max = int(xyxy[index - 1][3])
                    center_x = (x_max - x_min) / 2 + x_min
                    center_y = (y_max - y_min) / 2 + y_min
                    ROI = np.array([[center_x], [center_y]], np.float32)
                    kalman.correct(ROI)
                    prediction = tools.kalman_predict(kalman, ROI)
                    predicted_x = int(prediction[0][0])  # x-Koordinate
                    predicted_y = int(prediction[1][0])  # y-Koordinate
                else:
                    prediction = None

                if index == len(labels) - 1:
                    if prediction is not None:
                        cv2.circle(image_np, (predicted_x, predicted_y), radius=5, color=(0, 255, 0), thickness=-1)
                        print(f"Predicted position: ({predicted_x}, {predicted_y})")
                        overlay = utils.draw_image(image_np, masks, xyxy, scores, labels)
                    else:
                        overlay = utils.draw_image(image_np, masks, xyxy, scores, labels)

                else:
                    if prediction is not None:
                        cv2.circle(image_np, (predicted_x, predicted_y), radius=5, color=(0, 255, 0), thickness=-1)
                        print(f"Predicted position: ({predicted_x}, {predicted_y})")
                        image_np = utils.draw_image(image_np, masks, xyxy, scores, labels)
                    else:
                        image_np = utils.draw_image(image_np, masks, xyxy, scores, labels)
        return overlay, scores

def VideoInference(video_path, text_prompt, output_path, model, confidence_threshold, kalman):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Videoeigenschaften abrufen
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec für das Ausgabevideo
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    count = 0
    iteration_id = 1
    confidence_data = []
    while True:
        ret, frame = cap.read()

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        client.publish(topic_2, frame_bytes)

        if not ret:
            break

        if len(frame.shape) == 2:
            image_pil = Image.fromarray(frame, mode='L')
        else:
            image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        with torch.no_grad():
            overlay, scores = ImageInference(image_pil, text_prompt, start_frame, end_frame, model, confidence_threshold, kalman)
            iteration_id += 1
            confidence_data.extend([(iteration_id, score) for score in scores])

        overlay_bgr = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)

        out.write(overlay_bgr)
        del overlay_bgr, overlay, image_pil


        if count < 10:
            count += 1
        else:
            print("Clearing GPU cache...")
            torch.cuda.empty_cache()
            count = 0


        print(f"counted:{count}")
        frame_count += 1
        print(f"Processed frame {frame_count}")

    cap.release()
    out.release()
    print(f"Video saved to {output_path}")
    return confidence_data

if __name__ == "__main__":
    #MQTT Broker
    broker = 'emqx1.eqmx.io'
    port = 1883

    topic_1 = "bbox/topic"
    topic_2 = "frame/topic"

    client_id = f'python-mqtt-{random.randint(0, 1000)}'

    client = tools.mqtt_client.Client(client_id)

    client.connect(broker, port)

    client.loop_start()

    args = get_args_parser().parse_args()
    image_path = args.image_path
    text_prompt = args.text_prompt
    video_path = args.video_path
    output_path = args.output_path
    start_frame = args.start_frame
    end_frame = args.end_frame
    confidence_threshold = args.confidence_threshold
    model = LangSAM()
    kalman = tools.kalman_init()

    #Entscheidung Bild oder Video
    if video_path:
        output_path = f"{output_path}/output_video.mp4"
    else:
        output_path = f"{output_path}/output_image.png"

    #Entscheidung über Inferencemethode
    if video_path:
        confidence_data = VideoInference(video_path, text_prompt, output_path, model, confidence_threshold, kalman)
        plt.scatter(*zip(*confidence_data), marker='o', color='red')
        plt.title("Confidence Scores")
        plt.xlabel("Iteration")
        plt.ylabel("Confidence Score")
        plt.grid()
        plt.legend()
        plt.savefig(f"./output/confidence_scores_{confidence_threshold}.png")

    elif start_frame != 0 and end_frame is not None:
        overlay = ImageInference(image_path, text_prompt, start_frame, end_frame, model, confidence_threshold)

    else:
        image_pil = Image.open(image_path).convert("RGB")
        overlay = ImageInference(image_pil, text_prompt, start_frame, end_frame, model, confidence_threshold)