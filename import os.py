import os
import cv2
import numpy as np

def create_dataset(output_folder, num_classes=2, num_images_per_class=2):
    os.makedirs(output_folder, exist_ok=True)
    for class_idx in range(num_classes):
        class_folder = os.path.join(output_folder, f'class{class_idx}')
        os.makedirs(class_folder, exist_ok=True)
        for img_idx in range(num_images_per_class):
            # Create a random grayscale image
            image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
            image_path = os.path.join(class_folder, f'image{img_idx}.jpg')
            cv2.imwrite(image_path, image)
    print(f"Dataset created at {output_folder}")

create_dataset('./dataset/images', num_classes=2, num_images_per_class=2)
