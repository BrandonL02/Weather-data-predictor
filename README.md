# Environmental Health Risk Prediction

This project uses machine learning regression models to predict a **health risk score** from environmental condition data.

The project compares a baseline **Decision Tree Regressor** with **Random Forest** and **Gradient Boosting** models.

## Models Used

* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting Regressor

The ensemble models were optimized using:

* Grid Search
* Randomized Search
* `max_depth`
* `min_samples_leaf`

## Results

| Model                    | Mean Squared Error | R² Score |
| ------------------------ | -----------------: | -------: |
| Baseline Decision Tree   |             0.0197 |   0.9591 |
| Best Random Forest Model |             0.0113 |   0.9765 |

The best-performing model was the **Random Forest Regressor using Randomized Search with `max_depth` tuning**.

## Dataset

The project uses the following dataset:

`Environmental conditions dataset.xlsx`

The following columns were removed before model training:

* `datetimeEpoch`
* `month`
* `dayOfWeek`
* `isWeekend`

The remaining data contains:

* 30 feature columns
* 1 health risk score target column
* 80% training data
* 20% testing data

## Requirements

* Python
* pandas
* scikit-learn
* openpyxl

Install the required packages:

```bash
pip install pandas scikit-learn openpyxl
```

## How to Run

1. Place `Environmental conditions dataset.xlsx` in the same folder as the Python script.
2. Install the required packages.
3. Run the program:

```bash
python main.py
```

Replace `main.py` with the actual name of the Python file if needed.

## Evaluation

The models are evaluated using:

* **Mean Squared Error (MSE):** Lower values indicate better performance.
* **R² Score:** Values closer to 1 indicate better performance.

## Author

**Brandon Latimer**

B.S. Computer Science
