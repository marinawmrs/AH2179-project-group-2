# Predicting Bus Arrival Delay in Stockholm using Machine Learning

This repository includes the code used for completing the final project of the [Applied AI in Transport (AH2179)](https://github.com/zhenliangma/Applied-AI-in-Transportation/tree/main) course at KTH, during HT 2024.

### Project Description _[adapted from course contents]_

Public transport agencies have been long struggling to deliver core functions: frequent, reliable, and accessible collective mobility. The competition between public and private mobility services has come to light in recent years. The competition, accelerated by the COVID pandemic, results in declining transit ridership, budgetary fiscal cliffs for agencies to make ends meet, and evolving travel needs, norms, and options (e.g., working from home, micro-mobility, and on-demand services).

The agency believes that public transport can attract new and returning passengers by putting trust and experience first. Specifically, passengers need to be informed of the highest quality of real-time information (e.g., arrivals) and the most diverse range of information (e.g., alternative travel options)."

The aim of the project is to build machine learning models which predict the real-time delay at different bus stops for a selected route in Stockholm.


### Project Group 2
* [Kweku Abban](https://github.com/Kangkpe)
* [Joseph Ghareeb](https://github.com/josephg99)
* [Marina Wiemers](https://github.com/marinawmrs)


### Repository Structure
* ```data```: contains the dataset used in the project, taken from [here](https://github.com/zhenliangma/Applied-AI-in-Transportation/tree/main/ProjectAssignmentData)
* ```models```: contains the files for the trained models created in the project, as well as a subfolder containing stop-specific LR models
* ```results```: contains aggregated data of the evaluation metrics, paritioned into different domains, i.e. by stop sequence, scenario, and month
* ```1_preprocessing_feature_engineering.ipynb```,  ```Bus Arrival Data Exploration.pdf```, ```Delay_Exploration.py.py```, ```
PT_Project.ipynb```: These notebooks and files contain code and results obtained during the initial data exploration, and first attempts of simpler models.
* ```2_model_benchmarking_1000.ipynb```, ```2_model_benchmarking_545103.ipynb```: These notebooks contain the definition, training, and high-level evaluation of the different trained models, both for a smaller dataset (ca. 1/5th of the dataset), as well as the full one.
* ```3_model_eval.ipynb```: This notebook contains a more in-depth evaluation of the different models, among others paritioned into different scenarios of weather, time of day, and day of week.
* ```4_improving_lr.ipynb```: This notebook dives deeper into the selected, best-performing model and compares the performance of a general model to multiple, stop-specific ones. It also includes predictions for further ahead stops.


### Code Running Instructions
The indexed notebooks (```[X]_[notebook_name]```) are set up in such a way that allows them to be executed independently from each other, as they all access the same data and prepare it as needed. The different cells can simply be executed consecutively in e.g. Google Colab or Jupyter Notebook. 

*Note: In order to easily access the input dataset, as well as to load saved models, it is best to keep the same file structure.*
