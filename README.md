# Decision Tree Classifier

Decision tree classifier created on Python as a project for Intelligent Systems course. It uses CART algorithm with Gini impurity to find the best feature and threshold for a split. The decision tree model will create a json based on the tree and can be used using a parser provided.

## Announcement

Currently, this project only works for a specific dataset. Future updates will include compatibility with more types of dataset. This project may or not fit for your intended purposes, thus no guarantee is offered. Any changes to this project will be informed accordingly on this section.

### Files

- [Decision Tree Classifier Class](/decisiontreeclassifier.py)
- [Demo Notebook](/DecisionTreeClassifier_demo.ipynb)
- [Demo Parser](/decisiontreeparser.py)
- [Demo Json Tree Model](/decision_tree_model_demo.json)

### Dependencies

```
- Python3
- pandas
- json
```

### Usage

1. Install pandas on your Python.
2. Import DecisionTreeClassifier and pandas to the program.
3. Create a pandas dataframe using the dataset.
4. Split the dataset into training set (80%) and test set (20%).
5. Create a model by construction DecisionTreeClassifier object (ex: DecisionTreeClassifier(training_set, max_depth, min_sample_leaf) and a json file will be created based on the model.
6. Use the accuracy_test function to test the accuracy of the model.
7. Use the parser passing a dictionary with the same label as the dataset to predict.   

### TODO

1. Change the dataframe column iteration method to accomodate other dataset.
2. Finish documentation on the class.
3. Create a better readme file.

### Acknowledgement

Coming soon...