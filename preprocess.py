import anafora
import read_files as read
import os

#############################xml into raw data ############################################
#import anafora.timeml as timeml
# timeml._timeml_dir_to_anafora_dir("data/TBAQ-cleaned/AQUAINT/","data/TBAQ-cleaned/AQUAINT/")


############################  read xml from file  #########################################
# data = anafora.AnaforaData.from_file("ABC19980108.1830.0711/ABC19980108.1830.0711.TimeNorm.gold.completed.xml")
# for annotation in data.annotations:
#     annotation.spans
#     annotation.type

def get_xml_dir(dirname,format):
    '''
    get the directory for whole raw data and xml data, using the same root dir raw_text_dir
    :param raw_text_dir: root directory
    :return:  xml_data directory
    '''
    xml_file_dir = list()
    roots = os.listdir(dirname)
    root_folder = list()
    for root in roots:
        root_com =os.path.join(dirname,root)
        root_folder += [os.path.join(root_com,f) for f in os.listdir(root_com) if os.path.isdir(os.path.join(root_com, f))]
    for dir in root_folder:
        if format == "TimeML":
            xml_file_dir+= [os.path.join(dir,f) for f in os.listdir(dir) if f.endswith(".TimeNorm.gold.completed.xml")]
        elif format =="TimeNorm":
            xml_file_dir += [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(".TimeNorm.gold.completed.xml")]
    return xml_file_dir
    # for file in xml_file_dir:
    #     text_file = file.replace("data/TempEval-2013-Train","data/TBAQ-cleaned")
    #     text_file = text_file.replace(".TimeNorm.gold.completed.xml","")
    #     raw_data_dir.append(text_file)

#dirname = "data/TempEval-2013/"
def extrac_xmltag_anafora():
    # ##############time-ml ###############
    # desti = "data/time-ml/"
    # raw_dir_simple = read_from_json(desti + 'raw_dir_simple')
    # xml_dir =read_from_json(desti+'xml-timeml-file')
    # delete_annotation = ["TIMEX3"]
    # time_annotation = ["DATE","TIME","DURATION","SET"]
    ##############clinical_data ###############
    delete_annotation = ["Event","Modifier","PreAnnotation","NotNormalizable"]
    #raw_text_dir = read_from_json('clinical_data/train_samples1_simples')
    #xml_dir = read_from_json('clinical_data/xml_file_dir')

    #xmltags = list()
    for data_id in range(len(raw_dir_simple)):
        #print "\n"
        #print xml_dir[data_id]
        raw_text = readfrom_txt(raw_dir_simple[data_id])
        #xml_data_dir = dir.replace("TBAQ-cleaned",
        #                                            "TempEval-2013-Train") + ".TimeNorm.gold.completed.xml"
        ################# clinical_data ############################
        #data = anafora.AnaforaData.from_file("clinical_data/xml_files/" + raw_text_dir[data_id])
        ################ time_ml ###################################
        data = anafora.AnaforaData.from_file(xml_dir[data_id])

        posi_info_dict = dict()
        for annotation in data.annotations:
            #########anofora_norm #####################
            # if posi_info_dict.has_key(annotation.spans[0][0]):
            #     # posi_info_dict[annotation.spans[0][0]].append(annotation.spans[0][1])
            #     if annotation.type not in delete_annotation:
            #         posi_info_dict[annotation.spans[0][0]].append(annotation.type)
            #
            # else:
            #     anna_info = []
            #     terms = raw_text[annotation.spans[0][0]:annotation.spans[0][1]]
            #     anna_info.append(annotation.spans[0][1])
            #     anna_info.append(terms)
            #     anna_info.append(annotation.type)
            #     if annotation.type not in delete_annotation:
            #         posi_info_dict[annotation.spans[0][0]] = anna_info
            ######time_ml #######################
            if annotation.type in delete_annotation:
                property = annotation.properties._tag_to_property_xml
                if property.has_key("type"):
                    type = property["type"].text

                if posi_info_dict.has_key(annotation.spans[0][0]):
                    if annotation.type in delete_annotation:
                        posi_info_dict[annotation.spans[0][0]].append(type)

                else:
                    anna_info = []
                    print annotation.spans[0][0],annotation.spans[0][1]
                    terms = raw_text[annotation.spans[0][0]:annotation.spans[0][1]]
                    anna_info.append(annotation.spans[0][1])
                    anna_info.append(terms)
                    anna_info.append(type)
                    if type in time_annotation:
                        posi_info_dict[annotation.spans[0][0]] = anna_info
                    type = ''

            posi_info_dict = OrderedDict(sorted(posi_info_dict.items()))

        #print raw_dir_simple[data_id], "\n", posi_info_dict
        #print k
        #xmltags.append(posi_info_dict)
        save_in_json(raw_dir_simple[data_id]+"_tag_dict",posi_info_dict)

#extrac_xmltag()
