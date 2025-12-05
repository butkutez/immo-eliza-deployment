# Immo Eliza Price Predictor
[![forthebadge made-with-python](https://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![forthebadge deployed-with-streamlit](https://forthebadge.com/api/badges/generate?panels=2&primaryLabel=Deployed+with&secondaryLabel=Streamlit&primaryBGColor=%23ef4041&primaryTextColor=%23FFFFFF&secondaryBGColor=%23c1282d&secondaryTextColor=%23FFFFFF&primaryFontSize=12&primaryFontWeight=400&primaryLetterSpacing=2&primaryFontFamily=Roboto&primaryTextTransform=uppercase&secondaryFontSize=12&secondaryFontWeight=900&secondaryLetterSpacing=2&secondaryFontFamily=Montserrat&secondaryTextTransform=uppercase&secondaryIcon=streamlit&secondaryIconColor=%23FFFFFF&secondaryIconSize=16&secondaryIconPosition=right)](https://forthebadge.com)

[![Property_Price_Predictor](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*smf9XOCYX-7NuQSkE4m84g.jpeg)](https://medium.com/@varun.tyagi83/house-price-prediction-with-machine-learning-d49f93d681f2)
*Image source: [Medium](https://medium.com/@varun.tyagi83/house-price-prediction-with-machine-learning-d49f93d681f2)*



## **Project Description**
This project delivers a standalone web application for the Belgian real estate company Immo Eliza. The application is built using Streamlit and hosts a pre-trained machine learning regression model to provide non-technical employees and clients with instant property price estimations.

The application allows users to interactively input key property features (e.g., province, living area size, number of bedrooms) via a sidebar interface. The application loads the model and feature data locally, calculates the estimated price, and displays the result along with a confidence interval.


## **Getting started**

### Installation

**1. Clone the project**

```
cmd git clone https://github.com/butkutez/immo-eliza-deployment.git
```
**2. Navigate to the project folder**
```
cd immo-eliza-scraping
```
**3. Install required packages**

```
pip install -r requirements.txt
```

**4. Run the main scraper**

````
streamlit run app.py
````

## **Repo structure**

```
IMMO-ELIZA-DEPLOYMENT
├── .venv
│   └── .gitignore
├── app.py
├── final_cleaned_data.csv
├── Immo_Eliza_Predictor.png
├── model.pkl
├── README.md
└── requirements.txt
```

## **Usage**
The Streamlit application provides a simple, interactive user interface:

- <u>Input Features</u>: Use the sidebar to select and adjust the desired property features (province, property type/subtype, living area size, number of bedrooms, amenities like garden or pool etc.).

- <u>Prediction</u>: The price prediction and a confidence interval (based on a Log MAE of 0.1) are instantly displayed in the main panel.

- <u>Verification</u>: The "View Selected Features" expander allows users to confirm the exact inputs used for the prediction.

## **The core functionality involves:**

- Loading the model artifact (model.pkl) and feature options (final_cleaned_data.csv).

- Collecting user input into a single-row Pandas DataFrame.

- Calling the model's predict method.

- Applying the inverse log-transform (np.expm1) to convert the log-price prediction back to Euros for a meaningful result.

## **Result / Timeline**
The final application can be viewed here:*Image source: [Immo Eliza](https://immo-eliza-deployment-elhbdqkkclc7tvn5zybt5a.streamlit.app/)* 

This project was completed over 3 days, fulfilling the main objective of deploying a machine learning model via a Streamlit web application.

## **Personal Situation**
This project was completed as part of the Model Deployment module during the AI & Data Science Bootcamp at BeCode.org.

Connect with me on [LinkedIn](https://www.linkedin.com/in/zivile-butkute/).



[![Property Moving GIF](https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExbnMyZ3lrb2NzeWtpNXlxczZpMnFlbGFzY2FuMWZpYzlodGk2ZTRleiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/xUPGGuzpmG3jfeYWIg/giphy.gif)](https://giphy.com/gifs/house-home-moving-xUPGGuzpmG3jfeYWIg)
*Image source: [GIPHY](https://giphy.com/gifs/house-home-moving-xUPGGuzpmG3jfeYWIg)*