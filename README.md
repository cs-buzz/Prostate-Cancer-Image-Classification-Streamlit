# Prostate Cancer Classification with Deep Convolutional Neural Network

This is a 30-hour project for TC2013 Machine Learning. For full description and analysis please refer to Project Report.pdf.

## Web app
https://cs-buzz-prostate-cancer-classification-streamlit-app-soni3q.streamlit.app/

## Requirements
tensorflow: 2.8.0
keras: 2.6.0
streamlit: 1.8.1

## Results

| Models         |  Test   | Depth (layers) |  # Params  |
| -------------- | :-----: | :------------: | :--------: |
| Baseline Model |   30%   |       12       |  840,196   |
| ResNet50 V2    |   70%   |       6        | 24,615,940 |
| Inception V3   | **81%** |      234       | 47,666,084 |
| DenseNet 121   |   30%   |      607       | 12,268,548 |
| DenseNet 201   | **80%** |      707       | 18,321,984 |

## The Dataset
https://ukmedumy-my.sharepoint.com/personal/afzan_ukm_edu_my/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fafzan%5Fukm%5Fedu%5Fmy%2FDocuments%2FTC2013%20HUKM%20dataset%2Fprostate%20class
