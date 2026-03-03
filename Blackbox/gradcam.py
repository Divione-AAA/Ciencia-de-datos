import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

class gradCam():
    
    def obtener_ultima_capa(model):
        for layer in reversed(model.layers):
            if isinstance(layer,tf.keras.layers.Conv2D):
                return layer.name
        raise ValueError("No se encontro la capa")

    def creando_gradCam_model(model,nombre_ultima_capa):
        ultima_capa_convulsional = model.get_layer(nombre_ultima_capa)

        grad_model = tf.keras.models.Model(
            input = model.input,
            outputs = [ultima_capa_convulsional.output, model.output]
        )

        return grad_model
    
    def crear_heatmap(self,model, img_array, last_conv_layer_name, class_index=None):

        grad_model = self.creando_gradCam_model(model=model, nombre_ultima_capa=last_conv_layer_name)

        with tf.GradientTape() as tape:

            conv_outputs, predictions = grad_model(img_array)

            if class_index is None:
                class_index = tf.argmax(predictions[0])

            class_chanel = predictions[:, class_index]

        #Gradientes
        grads = tape.gradient(class_chanel,conv_outputs)

        #Promedio espacial
        pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

        #Multiplicar pesos por mapas
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # ReLU
        heatmap = tf.maximum(heatmap, 0)

        # Normalizar
        heatmap /= tf.reduce_max(heatmap)

        return heatmap.numpy()
    
    def superponer(img_path,heathmap,alpha=0.04):

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)

        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        superimposed_img = heatmap * alpha + img
        superimposed_img = np.uint8(superimposed_img)

        plt.imshow(superimposed_img)
        plt.axis("off")
        plt.show()

    def preprocess_image(img_path, target_size):

        img = tf.keras.preprocessing.image.load_img(
            img_path, target_size=target_size
        )
        
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        img_array = tf.keras.applications.vgg16.preprocess_input(img_array)

        return img_array
    
    def __init__(self):
        model = tf.keras.applications.vgg16(weights="imagenet")

        last_conv_layer_name = self.obtener_ultima_capa(model)

        img_path = "test.jpg"
        img_array = self.preprocess_image(img_path, (224, 224))

        heatmap = self.crear_heatmap(
            model,
            img_array,
            last_conv_layer_name
        )

        self.superponer(img_path, heatmap)