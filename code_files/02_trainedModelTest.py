# Evaluate test images stored in class folders, e.g., dataset/test/muffin.

from tensorflow.keras.models import load_model  # Load a saved Keras model.
import cv2  # Image operations.
import numpy as np  # Array operations.
import os  # File and folder operations.
import json



def testing_model(model_name):
    model_path = "./trained_models"

    # Load the model for inference; the filename is case-sensitive.
    model = load_model(model_path+f"/{model_name}.h5", compile=False)

    # Load labels in the same order as the model outputs.
    class_names = open(model_path+"/labels.txt", "r").readlines()

    # Remove each label's single-digit index and surrounding whitespace.
    for idx in range(len(class_names)):
        class_names[idx] = class_names[idx][2:].strip()

    # Class names must match the test folder names and model labels.
    test_dir = "./datasets/test"
    classes = ["good", "defect"]

    # Track the total and correct predictions for each true class.
    total_per_class   = {c: 0 for c in class_names}
    correct_per_class = {c: 0 for c in class_names}

    # Use each test folder's name as the ground-truth label.
    for true_label in classes:
        folder_path = os.path.join(test_dir, true_label)

        for filename in os.listdir(folder_path):
            # Skip files with unsupported extensions.
            if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            img_path = os.path.join(folder_path, filename)

            # Read the original BGR image; a valid image file is assumed.
            image_ = cv2.imread(img_path)

            # Resize to the model input size.
            image = cv2.resize(image_, (224, 224), interpolation=cv2.INTER_AREA)

            # Convert to float32 and add a batch dimension: (1, 224, 224, 3).
            image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

            # Scale pixels to [-1, 1]. The BGR order must match training.
            image = (image / 127.5) - 1

            # Predict class scores and select the highest-scoring class.
            prediction = model.predict(image)
            pred_index = np.argmax(prediction)
            pred_label = class_names[pred_index]
            confidence_score = prediction[0][pred_index]

            # Count all test images and correct predictions by true class.
            total_per_class[true_label] += 1
            if pred_label == true_label:
                correct_per_class[true_label] += 1


    # Report per-class accuracy; every class must contain test images.
    result_dict = {}
    for cls in class_names:
        acc = correct_per_class[cls] / total_per_class[cls] * 100
        result_dict[cls] = acc
    # Combine counts across all classes.
    total_correct = sum(correct_per_class.values())
    total_images  = sum(total_per_class.values())

    # Overall accuracy = correct predictions / total images * 100.
    overall_acc = total_correct / total_images * 100
    result_dict["Total_acc"] = overall_acc

    return result_dict

if __name__ == "__main__":

    # Test of each model and save as json file
    case_list = ["Q1_1", "Q1_2", "Q1_3", "Q2_1", "Q2_2", "Q3_1", "Q3_2", "Q3_3"]
    result_dict = {}

    for case in case_list:
        result_dict[case] = testing_model(case)
    
    with open("./results/acc_result.json", "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)