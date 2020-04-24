import json

# Recursively find the label based on threshold
def recursive_predict(data, branch):
    if isinstance(branch, dict):
        if data[branch['feature']] < branch['threshold']:
            return recursive_predict(data, branch['left'])
        else:
            return recursive_predict(data, branch['right'])
    else:
        return branch

# Load the json file of the decision tree model
def predict(data):
    file = open("decision_tree_model_demo.json", "r")
    model = json.loads(file.read())
    return recursive_predict(data, model)

if __name__ == "__main__":
    features = ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc',
                'four_g', 'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc',
                'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time',
                'three_g', 'touch_screen', 'wifi']
    values = [842,0,2.2,0,1,0,7,0.6,188,2,2,20,756,2549,9,7,19,0,0,1]
    data = {}
    for i in range(len(features)):
        data[features[i]] = values[i]
    prediction = predict(data)
    # Should print 1
    print(prediction)