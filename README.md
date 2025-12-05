# Immo Eliza Price Predictor
[![forthebadge made-with-python](https://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/) 
[![forthebadge deployed-with-streamlit](data:image/svg+xml;base64,PHN2ZyBkYXRhLXYtM2M4N2I3YjQ9IiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB3aWR0aD0iMjc1LjkzODE4NjY0NTUwNzgiIGhlaWdodD0iMzUiIHZpZXdCb3g9IjAgMCAyNzUuOTM4MTg2NjQ1NTA3OCAzNSIgY2xhc3M9ImJhZGdlLXN2ZyI+PGRlZnMgZGF0YS12LTNjODdiN2I0PSIiPjwhLS0tLT48IS0tLS0+PCEtLS0tPjwvZGVmcz48cmVjdCBkYXRhLXYtM2M4N2I3YjQ9IiIgd2lkdGg9IjEzNi42MjkxNTAzOTA2MjUiIGhlaWdodD0iMzUiIGZpbGw9IiNlZjQwNDEiLz48cmVjdCBkYXRhLXYtM2M4N2I3YjQ9IiIgeD0iMTM2LjYyOTE1MDM5MDYyNSIgd2lkdGg9IjEzOS4zMDkwMzYyNTQ4ODI4IiBoZWlnaHQ9IjM1IiBmaWxsPSIjYzEyODJkIi8+PCEtLS0tPjx0ZXh0IGRhdGEtdi0zYzg3YjdiND0iIiB4PSI2OC4zMTQ1NzUxOTUzMTI1IiB5PSIxNy41IiBkeT0iMC4zNWVtIiBmb250LXNpemU9IjEyIiBmb250LWZhbWlseT0iUm9ib3RvLCBzYW5zLXNlcmlmIiBmaWxsPSIjRkZGRkZGIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBsZXR0ZXItc3BhY2luZz0iMiIgZm9udC13ZWlnaHQ9IjQwMCIgZm9udC1zdHlsZT0ibm9ybWFsIiB0ZXh0LWRlY29yYXRpb249Im5vbmUiIGZpbGwtb3BhY2l0eT0iMSIgZm9udC12YXJpYW50PSJub3JtYWwiIHN0eWxlPSJ0ZXh0LXRyYW5zZm9ybTogdXBwZXJjYXNlOyI+REVQTE9ZRUQgV0lUSDwvdGV4dD48ZyBkYXRhLXYtM2M4N2I3YjQ9IiIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMjQ5LjkzODE4NjY0NTUwNzgsIDkuNSkgc2NhbGUoMC42NjY2NjY2NjY2NjY2NjY2KSI+PHBhdGggZGF0YS12LTNjODdiN2I0PSIiIGQ9Ik0xNi42NzMgMTEuMzJsNi44NjItMy42MThjLjIzMy0uMTM2LjU1NC4xMi40NDIuMzg3TDIwLjQ2MyAxNy4xem0tOC41NTYtLjIyOWwzLjQ3My01LjE4N2MuMjAzLS4zMjguNTc4LS4zMTYuNzkzLS4wMjhsNy44ODYgMTEuNzV6bS0zLjM3NSA3LjI1Yy0uMjggMC0uODM1LS4yODQtLjk5My0uNzE2bC0zLjcyLTkuNDZjLS4xMTgtLjMzMS4xMzktLjYxNC40OC0uNDY0bDE5LjQ3NCAxMC4zMDZjLS4xNDkuMTQ3LS40NTMuMzM3LS43Mi4zMzR6IiBmaWxsPSIjRkZGRkZGIi8+PC9nPjx0ZXh0IGRhdGEtdi0zYzg3YjdiND0iIiB4PSIxOTYuMjgzNjY4NTE4MDY2NCIgeT0iMTcuNSIgZHk9IjAuMzVlbSIgZm9udC1zaXplPSIxMiIgZm9udC1mYW1pbHk9Ik1vbnRzZXJyYXQsIHNhbnMtc2VyaWYiIGZpbGw9IiNGRkZGRkYiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtd2VpZ2h0PSI5MDAiIGxldHRlci1zcGFjaW5nPSIyIiBmb250LXN0eWxlPSJub3JtYWwiIHRleHQtZGVjb3JhdGlvbj0ibm9uZSIgZmlsbC1vcGFjaXR5PSIxIiBmb250LXZhcmlhbnQ9Im5vcm1hbCIgc3R5bGU9InRleHQtdHJhbnNmb3JtOiB1cHBlcmNhc2U7Ij5TVFJFQU1MSVQ8L3RleHQ+PCEtLS0tPjwvc3ZnPg==)](https://forthebadge.com)
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