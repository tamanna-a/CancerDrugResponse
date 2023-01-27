#define model building function with hyper parameter function hp as input
def build_model(hp):
  model = keras.Sequential()
  model.add(keras.layers.Dense(600, input_dim=600, activation='relu'))
  model.add(keras.layers.Dense(units=512, activation='relu'))
  #adds dropout layer- hyperparam- tries from 10 to 50%
  model.add(keras.layers.Dropout(
            hp.Float(
                'dropout',
                min_value=0.1,
                max_value=0.5,
                default=0.1,
                step=0.1)))
  for i in range(hp.Int('layers', 2,6)):
        model.add(keras.layers.Dense(units=hp.Int('units_' + str(i),
                                                  min_value=128, max_value=256,
                                                  step=32), activation='relu'))
        model.add(
            keras.layers.Dropout(
                hp.Float(
                    'dropout',
                    min_value=0.1,
                    max_value=0.5,
                    default=0.1,
                    step=0.1)
            )
        )

  model.add(keras.layers.Dense(1))
  hp_learning_rate = hp.Choice('learning_rate', values=lr_values)
  model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='mean_squared_error')
  return model
