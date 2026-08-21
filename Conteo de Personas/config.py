import tensorflow as tf

CONFIG = {
    "IMAGE_SIZE":640,
    "BATCH_SIZE":8,
    "BUFFER_SIZE":1000,
    "AUTOTUNE":tf.data.AUTOTUNE,
    "SEED":123,
    "GRID_SIZE":80,
    "NUM_CLASSES":1,
    "EPOCHS":30,
    "LEARNING_RATE":1e-4,
    "CHECKPOINT_DIR":"checkpoints",
    "CONFIDENCE_THRESHOLD":0.5,
    "NMS_IOU_THRESHOLD":0.5
}