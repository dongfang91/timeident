import os
import read_files as read
import numpy as np

def prob2classes_multiclasses( prediction):
    if prediction.shape[-1] > 1:
        return prediction.argmax(axis=-1)

def prob2classes_multiclasses_multioutput( prediction):
    output = list()
    for single_predic in prediction:
        if single_predic.shape[-1] > 1:
            output.append(single_predic.argmax(axis=-1))
    return output

def pro2classes_binaryclass(prediction):
    if prediction.shape[-1] <= 1:
        return (prediction > 0.5).astype('int32')

def make_prediction_function_multiclass(x_data,model_path,output_path,version = "0"):
    model1 = load_model(model_path)
    y_predict = model1.predict(x_data,batch_size=32)
    if len(y_predict)>=2:
        classes = prob2classes_multiclasses_multioutput(y_predict)
    else:
        classes = prob2classes_multiclasses(y_predict)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    np.save(output_path + "/y_predict_classes"+version, classes)
    read.save_in_pickle(output_path + "/y_predict_proba"+version, y_predict)

    return classes,y_predict

def hot_vectors2class_index (labels):
    examples = list()
    for instance in labels:
        label_index = list()
        for label in instance:
            if 1 in list(label):
                k = list(label).index(1)
                label_index.append(k)
            else:
                label_index.append(0)
        examples.append(label_index)
    return examples

def found_location_with_constraint(output):
    """
    :param output: the prediction sequences
    :return: a list of sentences with the span and tag identified
    """
    instance = list()
    instan_index = 0
    for instan in output:
        loc = list()
        for iter in range(len(instan)):
            #if not instan[iter] ==0 and iter <= instance_length[instan_index]-1:   #### with instance_length set
            if not instan[iter] == 0 :  #### without instance_length set
                loc.append([iter,instan[iter]])
        instance.append(loc)
        instan_index +=1
    return instance