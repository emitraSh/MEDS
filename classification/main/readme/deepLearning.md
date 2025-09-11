# How to check if model is overfitted?(Keras)
https://www.kaggle.com/code/ryanholbrook/overfitting-and-underfitting

Keras will keep a history of the training and validation loss over the epochs that it is training the model. by plotting the loss function and comparing it with the plot obtained with loss function of test data we can see if the weights given to model is valid or not.
- So, when a model learns signal both curves go down, but when it learns noise a gap is created in the curves. The size of the gap tells you how much noise the model has learned.
- validation loss plot will go down only when the model learns signal.

## when model is underfitted:
- One way is that you can increase the capacity of a network either by making it wider (more units to existing layers) or by making it deeper (adding more layers). Wider networks have an easier time learning more linear relationships, while deeper networks prefer more nonlinear ones. Which is better just depends on the dataset.
- Another way to use EarlyStopping in callback package

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=20, # how many epochs to wait before stopping
    restore_best_weights=True,
)
#These parameters say: "If there hasn't been at least an improvement of 0.001 in the validation loss over the previous 20 epochs,
# then stop the training and keep the best model you found."
```

- With neural networks, it's generally a good idea to put all of your data on a common scale, perhaps with something like scikit-learn's StandardScaler or MinMaxScaler.

- random.shuffle(training_data): The function random.shuffle(training_data) is used to randomly reorder the elements in the training_data list in-place. It is commonly used in machine learning or data processing tasks to eliminate bias during training.

# backpropagation
<img src="C:\Users\USER\PycharmProjects\classification\classification\main\readme\backpropagation.png"/>