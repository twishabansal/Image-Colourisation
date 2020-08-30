# Image Colourization :rainbow: 
[<img src="https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?&style=for-the-badge&logo=PyTorch&logoColor=white" />](https://pytorch.org/)
[<img src="https://img.shields.io/badge/heroku%20-%23430098.svg?&style=for-the-badge&logo=heroku&logoColor=white"/>](https://heroku.com/)

[![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)
[![GitHub contributors](https://img.shields.io/github/contributors/Naereen/StrapDown.js.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/contributors/)

An image colourisation model trained on 300k images using autoencoders to colourise grayscale images live [here](https://image-colouriser-streamlit.herokuapp.com/).

## Technology Stack
- PyTorch
- Streamlit
- Heroku
- Kaggle

## To Run the Notebook using Pretrained Weights

The path files for the models trained on landscapes, fruits, and clothes and people are available as **landscapes.pth**, **fruits.pth** and **clothes.pth**.

1. Clone the repository with `git clone https://gitlab.com/twishabansal/image-colourisation.git`
2. Open `image-colourization-starter.ipynb`.
3. To load a particular path file in your notebook, run -
```python
def load_checkpoint(filepath): 
    model = Encoder_Decoder()
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    
    return model
```
```
model = load_checkpoint(filepath)
```

## Train the Model from Scratch

Clone the repository with `git clone https://gitlab.com/twishabansal/image-colourisation.git`<br>
Documented Code for the model is available in the repository as `image-colourization-starter.ipynb` as an IPython notebook.<br>
Refer to the code written to process the data, define the model, train it, and finally get a prediction.

### Datasets 
The following datasets were used to train the model-
1. [ImageNet](https://www.kaggle.com/lijiyu/imagenet)
2. [Flickr](https://www.kaggle.com/hsankesara/flickr-image-dataset)
3. [Landscape Classification](https://www.kaggle.com/huseynguliyev/landscape-classification)
4. [Scene Classification](https://www.kaggle.com/nitishabharathi/scene-classification)
5. [Fruits360](https://www.kaggle.com/moltean/fruits)
6. [Fruit Recognition](https://www.kaggle.com/chrisfilo/fruit-recognition)
7. [Clothes Classification](https://www.kaggle.com/salil007/caavo)

## How To Run The Web App
1. Clone the repository with `https://github.com/Priyansi/image-colouriser-streamlit.git`
2. `pip install streamlit`
3. `streamlit run app/app.py` will run the app on `http://localhost:8501`

## References
1. [Colorful Image Colorization](https://arxiv.org/pdf/1603.08511.pdf)
2. [Applications of AutoEncoders - Image Colourisation](https://github.com/bnsreenu/python_for_microscopists)
<br>

[![forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)](https://forthebadge.com)