import read_files as read
import os
import output_transformation
import numpy as np

def span2xmlfiles(data_spans,file_name_simple):
    import anafora
    data = anafora.AnaforaData()
    id = 0
    for data_span in data_spans:
        e = anafora.AnaforaEntity()
        e.spans = ((int(data_span[0]), int(data_span[1])+1),)
        e.type = data_span[2]
        e.id = str(id)+"@e@" + file_name_simple
        data.annotations.append(e)
        id+=1
    #print data
    data.indent()
    #
    # outputfile = exp+ "\\"+raw_dir_simple[data_id]+"\\"
    # if not os.path.exists(outputfile):
    #     os.makedirs(outputfile)
    # data.to_file(outputfile+raw_dir_simple[data_id].replace(".txt","") +".TimeNorm.gold.completed.xml")
    return data

#span2xmlfiles()

def generate_output_multiclass(input_path,output_pred_path,pred =True,files_folder = "0",model_path = "")#version,dev="",target = "9_15",epoch = "02"):

    input = read.load_hdf5(input_path,["char","pos","unic"])

    if pred == True:
        classes,probs  = output_transformation.make_prediction_function_multiclass(input, model_path, output_pred_path, files_folder) #,,x_unic_onehot
    else:
        classes= np.load(output_pred_path+"/y_predict_classes"+files_folder+".npy")
        probs = read.readfrom_pickle(output_pred_path + "/y_predict_proba"+files_folder)

    spans = list()
    int2labels = list()

###############################evaluate character level performance on all input labels ##############

    for index in range(len(labels_name)):
    #index =3
        print "Character-level performance for ", labels_name[index]
        gold = output_transformation.hot_vectors2class_index(labels[index])
        class_loc = found_location_with_constraint(classes[index])
        gold_loc = found_location_with_constraint(gold)

        predictions = location2span(class_loc,probs[index],True)
        golds = location2span(gold_loc,probs[index],False)

        #test.calculate_precision_multi_class_new(class_loc, predictions, golds)
    ######################postprocess######################
        predictions_postprocess = post_process_dict(predictions, x_char,int2char)
        test.calculate_precision_multi_class_new(class_loc, predictions_postprocess,golds)

###########################################################################################################
