import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# --- 1. Data Loading and Preprocessing ---
IMG_SIZE = 128
NUM_CLASSES = 2 # Pet vs. Background

def preprocess_data(element):
    """
    Preprocesses an element from the Oxford-IIIT Pet dataset.
    Converts the mask to a binary 'pet' vs 'background' mask.
    """
    img = tf.image.resize(element['image'], (IMG_SIZE, IMG_SIZE))
    mask = tf.image.resize(element['segmentation_mask'], (IMG_SIZE, IMG_SIZE), method='nearest')
    
    # The pet dataset labels are: 1=Pet, 2=Background, 3=Outline.
    # We will map the Pet pixels to class 1, and everything else to class 0.
    mask = tf.where(mask == 1, 1, 0)

    img = tf.cast(img, tf.float32) / 255.0
    
    return img, mask

# Load the Oxford-IIIT Pet dataset
# This dataset is smaller and more reliable to download.
dataset, info = tfds.load('oxford_iiit_pet', with_info=True, split='train')

# For a real project, you'd use 'train' and 'test' splits.
# We'll split the training data for validation.
TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 64
BUFFER_SIZE = 1000

train_dataset = dataset.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# --- 2. Build the U-Net Model ---
# This U-Net model architecture is the same as before.
def unet_model(output_channels):
    inputs = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, 3])

    # Use a pre-trained model as the encoder
    base_model = tf.keras.applications.MobileNetV2(input_shape=[IMG_SIZE, IMG_SIZE, 3], include_top=False)

    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',      # 4x4
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    encoder = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
    encoder.trainable = False # Freeze the encoder

    # Decoder/Upsampler
    skips = reversed(encoder(inputs)[:-1])
    x = encoder(inputs)[-1]

    up_stack = [
        layers.Conv2DTranspose(filters, 3, strides=2, padding='same')
        for filters in [512, 256, 128, 64]
    ]

    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = layers.Conv2DTranspose(
        filters=output_channels,
        kernel_size=3,
        strides=2,
        padding='same',
        activation='softmax'
    )
    x = last(x)

    return models.Model(inputs=inputs, outputs=x)

model = unet_model(NUM_CLASSES)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# --- 3. Train the Model ---
EPOCHS = 20
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

history = model.fit(train_dataset, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH)

# Save the trained model
model.save('pet_segmentation_model.h5')
print("Model trained and saved as pet_segmentation_model.h5")