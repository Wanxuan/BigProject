from Version4.py import X_train, X_test, y_train, y_test

model = Sequential()

model.add(Conv2D(32, 3, 3, border_mode='same', init='he_normal', input_shape=train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(64, 3, 3, border_mode='same', init='he_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(128, 3, 3, border_mode='same', init='he_normal'))
model.add(MaxPooling2D(pool_size=(8, 8)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(10))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.Adam(lr=1e-3)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=128, nb_epoch=5, 
          verbose=1, validation_split=0.2, validation_data=(X_test, y_test))
