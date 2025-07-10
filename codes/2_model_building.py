model1 = Sequential()

# Add the first Convolutional block
model1.add(Conv2D(64, kernel_size=2, padding='same', activation='relu', input_shape=(48, 48, 3)))
model1.add(MaxPooling2D(pool_size=2))
model1.add(Dropout(0.2))

# Add the second Convolutional block
model1.add(Conv2D(32, kernel_size=2, padding='same', activation='relu'))
model1.add(MaxPooling2D(pool_size=2))
model1.add(Dropout(0.2))

# Add the third Convolutional block
model1.add(Conv2D(32, kernel_size=2, padding='same', activation='relu'))
model1.add(MaxPooling2D(pool_size=2))
model1.add(Dropout(0.2))

# Add the Flatten layer
model1.add(Flatten())

# Add the first Dense layer
model1.add(Dense(512, activation='relu'))
model1.add(Dropout(0.4))

# Add the Final layer
model1.add(Dense(4, activation='softmax'))

# Print model summary
# model1.summary()

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Define callbacks
checkpoint = ModelCheckpoint("./model1.h5",
                             monitor='val_accuracy',
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

# Compile the model
model1.compile(optimizer='adam',
               loss='categorical_crossentropy',
               metrics=['accuracy'])

# Train the model with callbacks
history1 = model1.fit(train_set,
                      epochs=20,
                      validation_data=validation_set,
                      callbacks=callbacks_list)


# Creating sequential model
model2 = Sequential()

# Add the first Convolutional block
model2.add(Conv2D(256, kernel_size=2, padding='same', activation='relu', input_shape=(48, 48, 3)))
model2.add(BatchNormalization())
model2.add(LeakyReLU(alpha=0.1))
model2.add(MaxPooling2D(pool_size=2))

# Add the second Convolutional block
model2.add(Conv2D(128, kernel_size=2, padding='same', activation='relu'))
model2.add(BatchNormalization())
model2.add(LeakyReLU(alpha=0.1))
model2.add(MaxPooling2D(pool_size=2))

# Add the third Convolutional block
model2.add(Conv2D(64, kernel_size=2, padding='same', activation='relu'))
model2.add(BatchNormalization())
model2.add(LeakyReLU(alpha=0.1))
model2.add(MaxPooling2D(pool_size=2))

# Add the fourth Convolutional block
model2.add(Conv2D(32, kernel_size=2, padding='same', activation='relu'))

# Add the Flatten layer
model2.add(Flatten())

# Adding the Dense layers
model2.add(Dense(512, activation='relu'))
model2.add(Dense(128, activation='relu'))
model2.add(Dense(4, activation='softmax'))


from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Define callbacks
checkpoint = ModelCheckpoint("./model2.h5",
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')  # monitor='val_loss' so mode should be 'min'

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
model2.compile(optimizer=Adam(learning_rate=0.001),
               loss='categorical_crossentropy',
               metrics=['accuracy'])

# Train the model
history2 = model2.fit(train_set,
                      epochs=20,
                      validation_data=validation_set,
                      callbacks=callbacks_list)


no_of_classes = 4

model3 = Sequential()

# Add 1st CNN Block
model3.add(Conv2D(64, kernel_size=2, padding='same', activation='relu', input_shape=(48, 48, 1)))
model3.add(BatchNormalization())
model3.add(LeakyReLU(alpha=0.1))
model3.add(MaxPooling2D(pool_size=2))
model3.add(Dropout(0.2))

# Add 2nd CNN Block
model3.add(Conv2D(128, kernel_size=2, padding='same', activation='relu'))
model3.add(BatchNormalization())
model3.add(LeakyReLU(alpha=0.1))
model3.add(MaxPooling2D(pool_size=2))
model3.add(Dropout(0.2))

# Add 3rd CNN Block
model3.add(Conv2D(512, kernel_size=2, padding='same', activation='relu'))
model3.add(BatchNormalization())
model3.add(LeakyReLU(alpha=0.1))
model3.add(MaxPooling2D(pool_size=2))
model3.add(Dropout(0.2))

# Add 4th CNN Block
model3.add(Conv2D(512, kernel_size=2, padding='same', activation='relu'))

# Add 5th CNN Block
model3.add(Conv2D(128, kernel_size=2, padding='same', activation='relu'))

# Flatten layer
model3.add(Flatten())

# First fully connected layer
model3.add(Dense(256))
model3.add(BatchNormalization())
model3.add(LeakyReLU(alpha=0.1))
model3.add(Dropout(0.3))

# Second fully connected layer
model3.add(Dense(512))
model3.add(BatchNormalization())
model3.add(LeakyReLU(alpha=0.1))
model3.add(Dropout(0.3))

# Final output layer
model3.add(Dense(no_of_classes, activation='softmax'))

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.optimizers import Adam

epochs = 35

steps_per_epoch = train_set.n // train_set.batch_size
validation_steps = validation_set.n // validation_set.batch_size

# Define callbacks
checkpoint = ModelCheckpoint("model3.weights.h5", monitor='val_accuracy',
                             save_weights_only=True, mode='max', verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2,
                              min_lr=0.0001, mode='auto')

csv_logger = CSVLogger('training_model3.log')

callbacks = [checkpoint, reduce_lr, csv_logger]

# Compile model3
model3.compile(optimizer=Adam(learning_rate=0.003),
               loss='categorical_crossentropy',
               metrics=['accuracy'])

# Fit model3
history = model3.fit(train_set,
                     epochs=epochs,
                     steps_per_epoch=steps_per_epoch,
                     validation_data=validation_set,
                     validation_steps=validation_steps,
                     callbacks=callbacks)