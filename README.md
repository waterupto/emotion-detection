# Emotion Detection using Deep Learning
This project aims to classify the emotion on a person's face into one of seven categories, using CNN's.
The dataset consists of 35889 48x48 sized face images with various emotions - fearful, angry, neutral, happy, sad, surprised and disgusted. 

## Dependencies

* Python 3, [OpenCV](https://opencv.org/), [Tensorflow](https://www.tensorflow.org/)
* To install the required packages, run `pip install -r requirements.txt`.

## Basic Usage

The repository is currently compatible with `tensorflow-2.0` and makes use of the Keras API using the `tensorflow.keras` library.

* First, clone the repository and enter the folder

```bash
git clone https://github.com/waterupto/emotion-detection.git
cd Emotion-detection
```

* If you want to train this model, use:  

```bash
cd src
python emotions.py --mode train
```

* If you want to view the predictions without training again, you can run the pre-trained model from here:  

```bash
cd src
python emotions.py --mode display
```

* The folder structure is of the form:  
  src:
  * data (folder)
  * `emotions.py` (file)
  * `haarcascade_frontalface_default.xml` (file)
  * `model.h5` (file)

* This implementation by default detects emotions on all faces in the webcam feed. With a simple 4-layer CNN, the test accuracy reached 63.2% in 50 epochs.

## Algorithm

* First, the **haar cascade** method is used to detect faces in each frame of the webcam feed.

* The region of image containing the face is resized to **48x48** and is passed as input to the CNN.

* The network outputs a list of **softmax scores** for the seven classes of emotions.

* The emotion with maximum score is displayed on the screen.

## Accuracy Plot 
![Accuracy plot](/accuracy.png)


## References

*"Deep-Emotion: Facial Expression Recognition Using Attentional Convolutional Network." Shervin Minaee, Amirali Abdolrashidi - University of California, Riverside*
