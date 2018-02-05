# Kaggle competition toxic-comments 

Implementation of a character-level LSTM model and data augmentation

Using Keras, I implemented a character-level LSTM model with character embeddings, convolutional layers, and LSTM layers.
The notebook runs the data augmentation and model fitting.

Since the original dataset contains comments that have many typos/misspellings, I also implemented a data augmentation method.
The data augmentation method will randomly add, remove, or replace characters in the string to simulate typos.

Notebook demonstrates the data processing and model training. 

## Observations
As of the date of writing this readme, the data augmentation technique and character-level model gave decent but not state of the art
results on Kaggle competition leaderboard. Few observations to note:

1. The character-level model does not seem to do as well as basic word-level model. Probably due to laack of training data
2. Adding more channels to the model can improve the model performance substantially. This is analogous to having more n-grams. The
notebook has example of such improvement.
3. Data augmentation technique improved the resutls slightly. Need more tuning of the augmentation probability to observe the effects.
4. The data augmentation techniques implemented here are rather naive. Probably can replace the augmentation technique with ones that
rely on more usual human error (e.g., replace "t" with "r" instead of just a random alphabet"
5. Character encoding is done at the senetence level. I tried encoding the whole comments or individual words but sentence level encoding returned the best performance on leaderboard.
