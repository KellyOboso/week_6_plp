import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('edge_ai_model')

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open('edge_ai_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Model converted to TensorFlow Lite!")
