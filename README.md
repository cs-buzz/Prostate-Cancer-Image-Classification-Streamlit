# Prostate Cancer Classification with Deep Convolutional Neural Network

This is a 30-hour project for TC2013 Machine Learning. For full description and analysis please refer to Project Report.pdf.

## Web app - Streamlit
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
https://ukmedumy-my.sharepoint.com/:f:/g/personal/afzan_ukm_edu_my/EvZX_tpa9WlBmq52vu0juisBmK6wNW45aIoxmFmaiw_Uew?e=BKDsCk
