import tensorflow as tf
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt

def display(display_list):
    """Displays a list of images side-by-side."""
    plt.figure(figsize=(15, 5))
    title = ['Input Image', 'Predicted Mask', 'Result']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

def create_mask(pred_mask):
    """Converts the model's output to a binary mask."""
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def apply_mask(image_path, background_path, model_path):
    """
    Loads an image, predicts the pet mask, and replaces the background.
    """
    IMG_SIZE = 128
    
    # 1. Load the trained model
    model = tf.keras.models.load_model(model_path)

    # 2. Load and preprocess the input image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_shape = img.shape
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)

    # 3. Predict the mask
    pred_mask_raw = model.predict(img_batch)
    mask = create_mask(pred_mask_raw)
    
    # 4. Resize mask to original image size
    mask_resized = cv2.resize(mask.numpy(), (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # 5. Load and resize background
    background = cv2.imread(background_path)
    if background is None:
        print(f"Error: Could not read background at {background_path}")
        return
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    background_resized = cv2.resize(background, (original_shape[1], original_shape[0]))

    # 6. Combine pet and background
    # Create a 3-channel mask for combining
    mask_3d = np.stack([mask_resized]*3, axis=-1)
    
    # Where mask is 1 (pet), use original image. Where it's 0, use background.
    output_image = np.where(mask_3d == 1, img, background_resized)

    # Save or display the result
    output_filename = 'pet_background_replaced.jpg'
    cv2.imwrite(output_filename, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
    print(f"Saved result to {output_filename}")

    # Display for quick check
    display([img, mask_3d * 255, output_image])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pet Segmentation Background Replacer")
    parser.add_argument('--image', type=str, required=True, help='Path to the input image of a pet.')
    parser.add_argument('--background', type=str, required=True, help='Path to the new background image.')
    parser.add_argument('--model', type=str, default='pet_segmentation_model.h5', help='Path to the trained .h5 model file.')
    
    args = parser.parse_args()
    
    apply_mask(args.image, args.background, args.model)