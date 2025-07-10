# Evaluate on test set
test_loss, test_accuracy = model1.evaluate(test_set)

print("Test Loss: {:.4f}".format(test_loss))
print("Test Accuracy: {:.4f}".format(test_accuracy))

#plot accuracy and loss curves

import matplotlib.pyplot as plt

# Plot accuracy
plt.figure(figsize=(8, 6))
plt.plot(history1.history['accuracy'], label='Training Accuracy')
plt.plot(history1.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot loss
plt.figure(figsize=(8, 6))
plt.plot(history1.history['loss'], label='Training Loss')
plt.plot(history1.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate model2 on test data
test_loss2, test_accuracy2 = model2.evaluate(test_set)

print("Test Loss (Model 2): {:.4f}".format(test_loss2))
print("Test Accuracy (Model 2): {:.4f}".format(test_accuracy2))


#accuracy and loss Curve
import matplotlib.pyplot as plt

# Plot accuracy
plt.figure(figsize=(8, 6))
plt.plot(history2.history['accuracy'], label='Training Accuracy')
plt.plot(history2.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model 2: Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot loss
plt.figure(figsize=(8, 6))
plt.plot(history2.history['loss'], label='Training Loss')
plt.plot(history2.history['val_loss'], label='Validation Loss')
plt.title('Model 2: Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

batch_size  = 32
img_size = 48

datagen_train = ImageDataGenerator(horizontal_flip = True,
                                   brightness_range = (0., 2.),
                                   rescale = 1./255,
                                   shear_range = 0.3)

train_set = datagen_train.flow_from_directory(folder_path + "train",
                                              target_size = (img_size, img_size),
                                              color_mode = 'rgb',
                                              batch_size = batch_size,
                                              class_mode = 'categorical',
                                              classes = ['happy', 'sad', 'neutral', 'surprise'],
                                              shuffle = True)

datagen_validation = ImageDataGenerator(rescale = 1./255)

validation_set = datagen_validation.flow_from_directory(folder_path + "validation",
                                                        target_size = (img_size, img_size),
                                                        color_mode = 'rgb',
                                                        batch_size = batch_size,
                                                        class_mode = 'categorical',
                                                        classes = ['happy', 'sad', 'neutral', 'surprise'],
                                                        shuffle = False)

datagen_test = ImageDataGenerator(rescale = 1./255)

test_set = datagen_test.flow_from_directory(folder_path + "test",
                                            target_size = (img_size, img_size),
                                            color_mode = 'rgb',
                                            batch_size = batch_size,
                                            class_mode = 'categorical',
                                            classes = ['happy', 'sad', 'neutral', 'surprise'],
                                            shuffle = False)


from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import Model

vgg = VGG16(include_top = False, weights = 'imagenet', input_shape = (48, 48, 3))

from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model

# Import VGG16 up to 'block5_pool' layer
vgg = VGG16(include_top=False, weights='imagenet', input_shape=(48, 48, 3))
vgg.trainable = False  # Freeze VGG16 layers

# Select output of 'block5_pool' layer
vgg_output = vgg.get_layer('block5_pool').output

# Add custom Fully Connected layers on top
x = Flatten()(vgg_output)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

# Final output layer
output = Dense(4, activation='softmax')(x)

# Build the model
model_vgg = Model(inputs=vgg.input, outputs=output)

pred = Dense(4, activation='softmax')(x)

# # Print model summary
# model_vgg.summary()


from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

vggmodel = Model(vgg.input, pred)

# Define callbacks
checkpoint = ModelCheckpoint("./vggmodel.h5",
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')

early_stopping = EarlyStopping(monitor='val_loss',
                               min_delta=0,
                               patience=3,
                               verbose=1,
                               restore_best_weights=True)

reduce_learningrate = ReduceLROnPlateau(monitor='val_loss',
                                        factor=0.2,
                                        patience=3,
                                        verbose=1,
                                        min_delta=0.0001)

callbacks_list = [early_stopping, checkpoint, reduce_learningrate]

# Compile the model
model_vgg.compile(optimizer=Adam(learning_rate=0.001),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

# Fit the model
history = model_vgg.fit(train_set,
                       epochs=20,
                       validation_data=validation_set,
                       callbacks=callbacks_list)


# Evaluate model_vgg on test data
test_loss_vgg, test_accuracy_vgg = model_vgg.evaluate(test_set)

print("Test Loss (VGG16 model): {:.4f}".format(test_loss_vgg))
print("Test Accuracy (VGG16 model): {:.4f}".format(test_accuracy_vgg))


#accuracy and loss curves

# accuracy
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('VGG16: Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('VGG16: Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# import tensorflow as tf
# import tensorflow.keras.applications as ap
# from tensorflow.keras import Model

# Resnet = ap.ResNet101(include_top = False, weights = "imagenet", input_shape=(48,48,3))
# Resnet.summary()

from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization

# Load ResNet101
Resnet = ap.ResNet101(include_top = False, weights = "imagenet", input_shape=(48,48,3))
# Resnet.summary()
# Select transfer layer
transfer_layer_Resnet = Resnet.get_layer('conv5_block3_add')
Resnet.trainable = False

# Add classification layers on top of it

# Flatten the output
x = Flatten()(transfer_layer_Resnet.output)
# Add a Dense layer with 256 neurons
x = Dense(256, activation='relu')(x)
# Add a Dense Layer with 128 neurons
x = Dense(128, activation='relu')(x)
# Add a DropOut layer with Drop out ratio of 0.3
x = Dropout(0.3)(x)
# Add a Dense Layer with 64 neurons
x = Dense(64, activation='relu')(x)
# Add a Batch Normalization layer
x = BatchNormalization()(x)
# Add the final dense layer with 4 neurons and use a 'softmax' activation
pred = Dense(4, activation='softmax')(x)
# Initialize the model (correct the input to use Resnet input, not vgg input)
resnetmodel = Model(Resnet.input, pred)


from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Define callbacks
checkpoint = ModelCheckpoint("./Resnetmodel.h5",
                             monitor='val_accuracy',   # small correction: 'val_accuracy' instead of 'val_acc'
                             verbose=1,
                             save_best_only=True,
                             mode='max')

early_stopping = EarlyStopping(monitor='val_loss',
                               min_delta=0,
                               patience=3,
                               verbose=1,
                               restore_best_weights=True)

reduce_learningrate = ReduceLROnPlateau(monitor='val_loss',
                                        factor=0.2,
                                        patience=3,
                                        verbose=1,
                                        min_delta=0.0001)

callbacks_list = [early_stopping, checkpoint, reduce_learningrate]

# 1. Compile your resnetmodel
resnetmodel.compile(optimizer=Adam(learning_rate=0.001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

# 2. Fit the model
history = resnetmodel.fit(train_set,
                          epochs=10,   # using 10 epochs as in your example
                          validation_data=validation_set,
                          callbacks=callbacks_list)


# Evaluate resnetmodel on test data
test_loss_resnet, test_accuracy_resnet = resnetmodel.evaluate(test_set)

print("Test Loss (ResNet model): {:.4f}".format(test_loss_resnet))
print("Test Accuracy (ResNet model): {:.4f}".format(test_accuracy_resnet))


# Plot accuracy
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('ResNet: Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('ResNet: Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

import tensorflow as tf
import tensorflow.keras.applications as ap
from tensorflow.keras import Model
EfficientNet = ap.EfficientNetV2B2(include_top=False,weights="imagenet", input_shape= (48, 48, 3))

# EfficientNet.summary()

from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization

# Load EfficientNetB0
EfficientNet = ap.EfficientNetB0(include_top=False, weights='imagenet', input_shape=(48, 48, 3))
# EfficientNet.summary()

# Select transfer layer
transfer_layer_EfficientNet = EfficientNet.get_layer('block6d_activation')
EfficientNet.trainable = False

# Add your Flatten layer
x = Flatten()(transfer_layer_EfficientNet.output)

# Add your Dense layers and/or BatchNormalization and Dropout layers
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)

# Add your final Dense layer with 4 neurons and softmax activation function
pred = Dense(4, activation='softmax')(x)

# Initialize the model
Efficientnetmodel = Model(EfficientNet.input, pred)


from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Define callbacks
checkpoint = ModelCheckpoint("./Efficientnetmodel.h5",
                             monitor='val_accuracy',  # correct to 'val_accuracy'
                             verbose=1,
                             save_best_only=True,
                             mode='max')

early_stopping = EarlyStopping(monitor='val_loss',
                               min_delta=0,
                               patience=3,
                               verbose=1,
                               restore_best_weights=True)

reduce_learningrate = ReduceLROnPlateau(monitor='val_loss',
                                        factor=0.2,
                                        patience=3,
                                        verbose=1,
                                        min_delta=0.0001)

callbacks_list = [early_stopping, checkpoint, reduce_learningrate]

# Compile your Efficientnetmodel
Efficientnetmodel.compile(optimizer=Adam(learning_rate=0.001),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

# Fit the model
history = Efficientnetmodel.fit(train_set,
                                epochs=10,
                                validation_data=validation_set,
                                callbacks=callbacks_list)


# Evaluate Efficientnetmodel on test data
test_loss_efficientnet, test_accuracy_efficientnet = Efficientnetmodel.evaluate(test_set)

print("Test Loss (EfficientNet model): {:.4f}".format(test_loss_efficientnet))
print("Test Accuracy (EfficientNet model): {:.4f}".format(test_accuracy_efficientnet))


# Plot accuracy
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('EfficientNet: Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('EfficientNet: Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


batch_size  = 32
img_size = 48

datagen_train = ImageDataGenerator(horizontal_flip = True,
                                   brightness_range = (0., 2.),
                                   rescale = 1./255,
                                   shear_range = 0.3)

train_set = datagen_train.flow_from_directory(folder_path + "train",
                                              target_size = (img_size, img_size),
                                              color_mode = 'grayscale',
                                              batch_size = batch_size,
                                              class_mode = 'categorical',
                                              classes = ['happy', 'sad', 'neutral', 'surprise'],
                                              shuffle = True)

datagen_validation = ImageDataGenerator(rescale = 1./255)

validation_set = datagen_validation.flow_from_directory(folder_path + "validation",
                                                        target_size = (img_size, img_size),
                                                        color_mode = 'grayscale',
                                                        batch_size = batch_size,
                                                        class_mode = 'categorical',
                                                        classes = ['happy', 'sad', 'neutral', 'surprise'],
                                                        shuffle = False)

datagen_test = ImageDataGenerator(rescale = 1./255)

test_set = datagen_test.flow_from_directory(folder_path + "test",
                                            target_size = (img_size, img_size),
                                            color_mode = 'grayscale',
                                            batch_size = batch_size,
                                            class_mode = 'categorical',
                                            classes = ['happy', 'sad', 'neutral', 'surprise'],
                                            shuffle = False)


# Evaluate model3 on test data
test_loss_model3, test_accuracy_model3 = model3.evaluate(test_set)

print("Test Loss (Model 3 - Complex CNN): {:.4f}".format(test_loss_model3))
print("Test Accuracy (Model 3 - Complex CNN): {:.4f}".format(test_accuracy_model3))


# Plot training and validation accuracy
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model 3 (Complex CNN): Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model 3 (Complex CNN): Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Re-create test_set with grayscale mode
test_set = datagen_test.flow_from_directory(folder_path + "test",
                                            target_size = (img_size, img_size),
                                            color_mode = 'grayscale',
                                            batch_size = 128,
                                            class_mode = 'categorical',
                                            classes = ['happy', 'sad', 'neutral', 'surprise'],
                                            shuffle = True)

# Get a batch of test images and labels
test_images, test_labels = next(test_set)

# Write the name of your chosen model
pred = model3.predict(test_images)
pred = np.argmax(pred, axis=1)
y_true = np.argmax(test_labels, axis=1)

# Printing the classification report
print(classification_report(y_true, pred, target_names=['happy', 'sad', 'neutral', 'surprise']))

# Plotting the heatmap using confusion matrix
cm = confusion_matrix(y_true, pred)
plt.figure(figsize=(8, 5))
sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues',
            xticklabels=['happy', 'sad', 'neutral', 'surprise'],
            yticklabels=['happy', 'sad', 'neutral', 'surprise'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix - Model 3 (Complex CNN)')
plt.show()

pip install nbconvert