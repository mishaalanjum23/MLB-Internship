from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

#load model
pretrained_model = MobileNetV2(weights = "imagenet", include_top = False, input_shape = (224, 224, 3))

#Explore
pretrained_model.summary()

#freeze layers
pretrained_model.trainable = False

#add classifier
model = Sequential([pretrained_model,
                  GlobalAveragePooling2D(),
                  Dense(128, activation="relu"),
                  Dense(3, activation="softmax")])