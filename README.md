# Event Registration Forecasting Model

## Overview
This Python-based project develops a predictive model for forecasting future event registrations, achieving an accuracy of 78.57%. It employs advanced statistical analysis to dissect historical data, distinguishing natural registration trends from those influenced by marketing efforts. The project includes a comprehensive data preparation and preprocessing pipeline, gradient calculation for trend identification, and a user-friendly graphical user interface (GUI) for interactive forecasting.

## Features
- **Data Preparation and Preprocessing:** Custom Python scripts for cleaning and structuring raw event registration data.
- **Gradient and Natural Growth Analysis:** Analysis technique to identify segments of data representing organic growth.
- **Forecasting Model:** Utilizes statistical methods to predict future registrations based on historical trends, excluding marketing impacts.
- **GUI for Forecasting:** Developed using Python's tkinter library, allowing users to input current registration numbers and days left in the registration period for an estimated final count.

## Technologies Used
- Python 3.10
- Pandas for data manipulation
- Matplotlib for data visualisation
- Tkinter for GUI development
- Linear and logistic regression for statistical modeling
  
## Project Structure
- **Data_filtering.py:** Script for data cleaning and preprocessing.
- **Curve_Segmentation:** Core script containing the logic for the forecasting model.
- **Gradient_Calculation.py:** Script for performing gradient and natural growth analysis.
- **Final_Gradient_Calculation.py:** Main Script for registration rate and error calculation.
- **GUI.py:** Script for the graphical user interface.

## Setup and Installation
Ensure you have Python 3.10 installed on your machine. Clone the repository and install the required dependencies:

git clone https://github.com/yourusername/event-registration-forecasting.git
cd event-registration-forecasting
pip install -r requirements.txt

## Usage
To run the forecasting model and interact with the GUI:

python gui.py

Input the current number of registrations and the remaining days in the registration period to receive the forecasted final registration count.

## Challenges and Learnings
This project navigated through challenges like data inconsistencies and the impact of the COVID-19 pandemic on registration behaviors, requiring iterative model refinements. It underscored the importance of continuous data analysis and model adaptation to evolving trends.

## Future Enhancements
- Expand the dataset to encompass a broader range of events.
- Refine the methodology to improve accuracy and utility.
- Integrate machine learning techniques for more dynamic forecasting.
