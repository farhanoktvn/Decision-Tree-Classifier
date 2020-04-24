import json
import pandas

class DecisionTreeClassifier:
    # Tree initialization
    def __init__(self, dataset, max_depth, min_size):
        self.tree = self.build_decision_tree(dataset, max_depth, min_size)
    
    # Splitting the data according to the threshold
    def check_split(self, dataset, feature, threshold):
        """
        If the value of the given feature is less than the threshold,
        it will be the left node. If it is greater than or equal to the 
        threshold, it will be the right node.

        >>> check_split(dataset, feature, threshold)
        Pandas.DataFrame
        """
        left_node, right_node = None, None
        left = dataset.loc[dataset[feature] < threshold]
        if len(left):
            left_node = left
        else:
            return None
        right = dataset.loc[dataset[feature] >= threshold]
        if len(right):
            right_node = right
        else:
            return None
        return left_node, right_node

    # Find the gini index
    def find_gini_index(self, sliced_df, temp_labels):
        """
        It will return the gini index if the dataset is split by the given
        label (temp_labels).

        >>> find_gini_index(sliced_df, temp_labels)
        0.1827
        """
        num_of_data = float(sum([len(slice) for slice in sliced_df]))
        gini_index = 0.0
        for slice in sliced_df:
            size = float(len(slice))
            if size == 0:
              continue
            temp_score = 0.0
            for label in temp_labels:
                # TODO change price_range to the last name of the column
                p = (slice.price_range == label).sum() / size
                temp_score += p**2
            gini_index += (1.0 - temp_score) * (size / num_of_data)
        return gini_index

    # Creating the end of branch for the tree
    def end_of_branch(self, dataset):
        """
        When the remaining dataset can not be further splitted or the depth 
        of the tree is reaching maximum, the end of branch will be created 
        consisting of a possible label.

        >>> end_of_branch(dataset)
        label
        """
        count = 0
        possible_label = None
        # TODO change price_range to the last name of the column
        for label in dataset.price_range.unique().tolist():
            currentCount = len([x for x in dataset.price_range == label if x == True])
            if currentCount > count:
                count = currentCount
                possible_label = label
        return possible_label

    # Finding the best split based on the Gini Index
    def best_split(self, dataset):
        """
        This function will iterate through all the labels, and all the values 
        of that label to find an optimum split with the smallest Gini Index.

        When end of branch condition is satisfied
        >>> best_split(dataset)
        label

        When a new split is found
        >>> best_split(dataset)
        {'feature': feature, 'value': threshold, 'sliced_df': sliced_df}
        """
        # TODO change price_range to the last name of the column
        temp_labels = list(set(dataset.price_range.values.tolist()))
        feature, threshold, gini_index, sliced_df = None, 0.0, 1, None
        # TODO change wifi to the end of feature column
        for col in dataset.loc[:,:'wifi'].columns.values.tolist():
            for val in dataset[col].unique():
                temp_sliced_df = self.check_split(dataset, col, val)
                if temp_sliced_df == None:
                    continue
                temp_gini_index = self.find_gini_index(temp_sliced_df, temp_labels)
                if temp_gini_index < gini_index:
                    feature = col
                    threshold = val
                    gini_index = temp_gini_index
                    sliced_df = temp_sliced_df
                else:
                  continue
        if feature == None:
            this_label = self.end_of_branch(dataset)
            return this_label
        return {'feature': feature, 'value': threshold, 'sliced_df': sliced_df}

    # Method to recursively split node
    def split_node(self, node, max_depth, min_size, current_depth):
        """
        Checking whether the conditions of end of branch is satisfied or 
        the node can be further splitted.

        >>> split_node(node, max_depth, min_size, current_depth)
        None
        """
        if isinstance(node, int):
            return
        left, right = node['sliced_df']
        del(node['sliced_df'])
        if left.empty or right.empty:
            node['left'] = node['right'] = self.end_of_branch(left + right)
            return
        if current_depth >= max_depth:
            node['left'], node['right'] = self.end_of_branch(left), self.end_of_branch(right)
            return
        if len(left) <= min_size:
            node['left'] = self.end_of_branch(left)
        else:
            node['left'] = self.best_split(left)
            self.split_node(node['left'], max_depth, min_size, current_depth+1)
        if len(right) <= min_size:
            node['right'] = self.end_of_branch(right)
        else:
            node['right'] = self.best_split(right)
            self.split_node(node['right'], max_depth, min_size, current_depth+1)

    # Building the decision tree based on the requirements
    def build_decision_tree(self, dataset, max_depth, min_size):
        """
        >>> build_decision_tree(dataset, max_depth, min_size)
        {'feature': feature, 'value': threshold, 'sliced_df': sliced_df}
        """
        root = self.best_split(dataset)
        self.split_node(root, max_depth, min_size, 1)
        self.create_json(root)
        return root

    # Traversing the tree based on the feature and threshold of each node
    def recursive_predict(self, node, data_row):
        """
        >>> recursive_predict(node, data_row)
        label
        """
        if data_row[node['feature']].unique()[0] < node['value']:
            if isinstance(node['left'], dict):
                return self.recursive_predict(node['left'], data_row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.recursive_predict(node['right'], data_row)
            else:
                return node['right']

    # Initiating the prediction
    def predict(self, data_row):
        node = self.tree
        return self.recursive_predict(node, data_row)
    
    # Testing the accuracy of trained model
    def accuracy_test(self, data, result):
        """
        For each row in the testing set, each one will be predicted.
        True/false prediction will be counted towards the result.

        >>> accuracy_test(data, result)
        0.76
        """
        indexes = len(data.index.values.tolist())
        true_pred = 0
        false_pred = 0
        for i in range(indexes):
            prediction = self.predict(data.iloc[[i]])
            # TODO change price_range to the last name of the column
            actual = result.iloc[[i]]['price_range'].values[0]
            if prediction == actual:
                true_pred += 1
                continue
            false_pred += 1
        return true_pred / (true_pred+false_pred)
    
    # Recursively make a dictionary to save the feature and threshold of nodes
    def recursive_tree_traverse(self, node):
        if isinstance(node, dict):
            return {'feature': node['feature'],
                    'threshold': int(node['value']),
                    'left': self.recursive_tree_traverse(node['left']),
                    'right': self.recursive_tree_traverse(node['right'])}
        else:
            return int(node)

    # Create a json based on the tree dictionary
    def create_json(self, root):
        self.tree_dict = self.recursive_tree_traverse(root)
        with open("decision_tree_model.json", "w") as write_file:
            json.dump(self.tree_dict, write_file)