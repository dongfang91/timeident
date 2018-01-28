# encoding: utf-8
import nlp_functionality as process
import anafora
import read_files as read
import os
from collections import OrderedDict
from nltk.tokenize import sent_tokenize
from nltk.tokenize.util import regexp_span_tokenize
import numpy as np
from collections import defaultdict

#############################processing time-ml file into raw data and anofora format ############################################
# import anafora.timeml as timeml
 # timeml._timeml_dir_to_anafora_dir("data/TBAQ-cleaned/","data/Processed",schema_name="TimeML")

############################  read xml from file  #########################################
# data = anafora.AnaforaData.from_file("ABC19980108.1830.0711/ABC19980108.1830.0711.TimeNorm.gold.completed.xml")
# for annotation in data.annotations:
#     annotation.spans
#     annotation.type

def get_xml_dir(read_dirname,output_folder,format):
    '''
    get the directory for whole raw data and xml data, using the same root dir raw_text_dir
    :param raw_text_dir: root directory
    :return:  xml_data directory
    '''
    xml_file_dir = list()
    raw_data_dir = list()
    roots = os.listdir(read_dirname)
    root_folder = list()
    for root in roots:
        root_com =os.path.join(read_dirname,root)
        root_folder += [os.path.join(root_com,f) for f in os.listdir(root_com) if os.path.isdir(os.path.join(root_com, f))]
    for dir in root_folder:
        for f in os.listdir(dir):
                if f.endswith("."+format+".gold.completed.xml"):
                    xml_file_dir_file = os.path.join(dir,f)
                    xml_file_dir.append(xml_file_dir_file)
                    text_file = xml_file_dir_file.replace(read_dirname, output_folder)
                    text_file = text_file.replace("."+format+".gold.completed.xml", "")
                    raw_data_dir.append(text_file)
    return xml_file_dir,raw_data_dir

#dirname = "data/TempEval-2013/"

def extract_xmltag_timeml(xml_file_dir,raw_text):
    #annotation = ["TIMEX3"]
    time_annotation = ["DATE","TIME","DURATION","SET"]
    data = anafora.AnaforaData.from_file(xml_file_dir)
    posi_info_dict = dict()
    for annotation in data.annotations:
        property = annotation.properties._tag_to_property_xml
        if property.has_key("type"):
            type = property["type"].text
            if type in time_annotation:
                process.addannotation_to_dict(posi_info_dict,annotation,raw_text)
    posi_info_dict = OrderedDict(sorted(posi_info_dict.items()))
    return posi_info_dict

def extract_xmltag_anafora(xml_file_dir,raw_text):
    delete_annotation = ["Event","Modifier","PreAnnotation","NotNormalizable"]
    data = anafora.AnaforaData.from_file(xml_file_dir)
    posi_info_dict = dict()
    for annotation in data.annotations:
        if annotation.type not in delete_annotation:
            posi_info_dict = process.addannotation_to_dict(posi_info_dict,annotation,raw_text)
    posi_info_dict = OrderedDict(sorted(posi_info_dict.items()))
    return posi_info_dict

##text_normalize
def split_by_sentence(raw_text,char_vocab):
    sent_tokenize_list = sent_tokenize(raw_text)
    sent_tokenize_span_list = process.spans(sent_tokenize_list, raw_text)
    sent_span_list = list()
    max_len = list()
    for sent_tokenize_span in sent_tokenize_span_list:
        sent_spans = list(regexp_span_tokenize(sent_tokenize_span[0], r'\n'))
        for sent_span in sent_spans:
            sent_span = (sent_span[0] + sent_tokenize_span[1], sent_span[1] + sent_tokenize_span[1])
            sent = raw_text[sent_span[0]:sent_span[1]]
            for char in sent:
                char_vocab[char]+=1
            if len(sent) >= 350:
                #print sent
                multi_sent_span, multi_sent_len = process.rule_based_tokenizer(sent, sent_span)
                sent_span_list += multi_sent_span
                max_len += multi_sent_len
                if max(multi_sent_len)>350:
                    print sent
            else:
                sent_span_list.append([sent, sent_span[0], sent_span[1]])
                max_len.append(len(sent))
    return sent_span_list,max_len,char_vocab

        #read.save_in_json(raw_dir_simple[data_id] + "_sent", sent_span_list)
        #max_len.sort(reverse=True)

def xml_tag_in_sentence(sentences,posi_info_dict):
    tag_list = list()
    tag_span = posi_info_dict.keys()
    tag_span = sorted(tag_span, key=int)
    i = 0
    for sent in sentences:
        tag = list()
        if i < len(tag_span):
            if sent[2] < int(tag_span[i]):
                tag_list.append(tag)
            elif sent[1] <= int(tag_span[i]) and sent[2] > int(tag_span[i]):
                while True:
                    tag.append((tag_span[i],posi_info_dict[tag_span[i]]))
                    i = i + 1
                    if i < len(tag_span):
                        if int(tag_span[i]) > sent[2]:
                            tag_list.append(tag)
                            break
                    else:
                        tag_list.append(tag)
                        break
                else:
                    tag_list.append(tag)
        #print tag_list
    return tag_list

def get_idx_from_sent(padding_char,sent, word_idx_map, max_l,pad):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []

    for i in xrange(pad):
        x.append(word_idx_map[padding_char])

    for word in sent:
        if word in word_idx_map.keys():
            x.append(word_idx_map[word])
        else:
            x.append(word_idx_map["unknown"])

    for i in xrange(pad):
        x.append(word_idx_map[padding_char])

    while len(x) < max_l+ 2 *pad:
        x.append(0)
    return x

def document_level_2_sentence_level(xml_file_dir,raw_data_dir,save_data_dir,max_len_file_name):

    max_len_all=list()

    char_vocab = defaultdict(float)
    pos_vocab = defaultdict(float)
    unicode_vocab = defaultdict(float)

    for data_id in range(0, len(xml_file_dir)):
        raw_text = read.readfrom_txt(raw_data_dir[data_id])
        raw_text = process.text_normalize(raw_text)
        sent_span_list_file, max_len_file,char_vocab = split_by_sentence(raw_text,char_vocab)
        # for max_sent_len in max_len_file:
        #     if max_sent_len >=350:
        #         print raw_data_dir[data_id]
        max_len_all +=max_len_file
        posi_info_dict = extract_xmltag_anafora(xml_file_dir[data_id],raw_text)
        sent_tag_list_file = xml_tag_in_sentence(sent_span_list_file,posi_info_dict)
        pos_sentences, pos_vocab = process.get_pos_sentence(sent_span_list_file, pos_vocab)
        pos_sentences_character = process.word_pos_2_character_pos(sent_span_list_file, pos_sentences)
        unico_sentences_characte,unicode_vocab = process.get_unicode(sent_span_list_file,unicode_vocab)
        read.savein_json(save_data_dir[data_id]+"_sent",sent_span_list_file)
        read.savein_json(save_data_dir[data_id]+"_tag",sent_tag_list_file)
        read.savein_json(save_data_dir[data_id] + "_pos", pos_sentences_character)
        read.savein_json(save_data_dir[data_id] + "_unicodecategory", unico_sentences_characte)
    max_len_all.sort(reverse=True)
    read.savein_json(max_len_file_name, max_len_all)

####Newswire Dataset #######
# dirname = "data/TempEval-2013/"
# output_folder = "data/Processed_TempEval/"
#max_len_file_name = output_folder +"max_len"
# xml_file_dir, raw_data_dir =get_xml_dir(dirname,output_folder,format)
#document_level_2_sentence_level(xml_file_dir, raw_data_dir,raw_data_dir,max_len_file_name)

####Thyme_Colon Dataset #####
text = read.textfile2list("data/test_file.txt")
output_folder = "data/Processed_THYMEColonFinal/"
max_len_file_name = output_folder +"max_len"
xml_file_dir = ["data/"+file +"/"+file.split('/')[-1]+".TimeNorm.gold.completed.xml" for file in text if "THYMEColonFinal" in file]
raw_data_dir = ["data/"+file +"/"+file.split('/')[-1] for file in text if "THYMEColonFinal" in file]
save_data_dir = ["data/"+(file +"/"+file.split('/')[-1]).replace("THYMEColonFinal","Processed_THYMEColonFinal") for file in text if "THYMEColonFinal" in file]

print xml_file_dir
print raw_data_dir
document_level_2_sentence_level(xml_file_dir, raw_data_dir,save_data_dir,max_len_file_name)


#document_level_2_sentence_level(xml_file_dir, raw_data_dir,max_len_file_name)




def features_extraction(raw_data_dir,mode = "train"):
    dirname = "data/TempEval-2013/"
    max_len = 350
    pad = 3
    input_char = list()
    input_pos = list()
    input_unic = list()
    char2int = read.readfrom_json("data/config_data/vocab/char2int")
    pos2int = read.readfrom_json("data/config_data/vocab/pos2int")
    unicode2int = read.readfrom_json("data/config_data/vocab/unicate2int")
    for data_id in range(0, len(raw_data_dir)):
        sent_span_list_file = read.readfrom_json(raw_data_dir[data_id] + "_sent")
        pos_sentences_character = read.readfrom_json(raw_data_dir[data_id] + "_pos")
        unico_sentences_characte = read.readfrom_json(raw_data_dir[data_id] + "_unicodecategory")
        n_sent = len(sent_span_list_file)
        for index in range(n_sent):
            input_char.append(get_idx_from_sent("\n",sent_span_list_file[index][0], char2int, max_len,pad))
            input_pos.append(get_idx_from_sent("\n",pos_sentences_character[index], pos2int, max_len,pad))
            input_unic.append(get_idx_from_sent("Cc",unico_sentences_characte[index], unicode2int, max_len,pad))

    input_char = np.asarray(input_char, dtype="int")
    input_pos = np.asarray(input_pos, dtype="int")
    input_unic = np.asarray(input_unic, dtype="int")

    read.save_hdf5("data/Processed_TempEval/train_input", ["char","pos","unic"], [input_char,input_pos,input_unic], ['int8','int8','int8'])

#def output_encoding():








