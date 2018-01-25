# encoding: utf-8
import re
from nltk.tokenize import sent_tokenize

def addannotation_to_dict(posi_info_dict,annotation,raw_text):
    if posi_info_dict.has_key(annotation.spans[0][0]):
            posi_info_dict[annotation.spans[0][0]].append(annotation.type)
    else:
        anna_info = []
        #print annotation.spans[0][0], annotation.spans[0][1]
        terms = raw_text[annotation.spans[0][0]:annotation.spans[0][1]]
        anna_info.append(annotation.spans[0][1])
        anna_info.append(terms)
        anna_info.append(annotation.type)
        posi_info_dict[annotation.spans[0][0]] = anna_info
    return posi_info_dict

def text_normalize(rawtext):
    # rawtext = rawtext.replace("•", "\n")
    # rawtext = rawtext.replace("’", "\'")
    # rawtext = rawtext.replace("”", "\"")
    # rawtext = rawtext.replace("“", "\"")
    # rawtext = rawtext.replace("‘", "\'")
    # rawtext = rawtext.replace("—", "-")
    # rawtext = rawtext.replace("¢", "c")
    # rawtext = rawtext.replace("°", "\'")
    # rawtext = rawtext.replace("%", "%")
    # rawtext = rawtext.replace("$", "$")
    return rawtext

def spans(sents,txt):
    sentence_chunkings = list()
    offset = 0
    for sent in sents:
        offset = txt.find(sent, offset)
        item = (sent, offset, offset + len(sent))
        offset += len(sent)
        sentence_chunkings.append(item)
    return sentence_chunkings

def add_start_end(sent_tokenize_span_list,start):
    sent_tokenize_span_list_new = list()
    max_len = list()
    for (sent,sent_start,sent_end) in sent_tokenize_span_list:
        sent_tokenize_span_list_new.append((sent,sent_start+start,sent_end+start))
        max_len.append(sent_end-sent_start)
    return sent_tokenize_span_list_new,max_len

def rule_based_tokenizer(sent,sent_span): # sent,sent_span

    #sent = "@ % This @ Oct 25 Oct 24 Year @ U.S. ................... 315.2 316.4 +23.1 @ Britain ................ 646.4 643.1 +18.4 @ Canada ................. 426.9 426.4 +16.3 @ Japan .................. 1547.1 1550.9 + 8.9 @ France ................. 518.6 521.2 +17.1 @ Germany ................ 236.7 241.0 +13.8 @ Hong Kong .............. 2049.2 2068.9 + 1.0 @ Switzerland ............ 212.6 216.5 +23.0 @ Australia .............. 326.0 329.4 +12.3 @ World index ............ 532.4 533.4 + 7.7 @ Weekly Percentage Leaders"
    #sent = "U.S. Attorney Denise E. O'Donnell declined to discuss what federal charges were being pursued, but she said that in a case like this, potential charges would be abortion-related violence, the use of a firearm in an act of violence, crossing state lines to commit a crime, and, if the suspect's act was tied to an organization, violation of the so-called RICO statutes, which prohibit an organized criminal enterprise."
    #sent = "WASHINGTON _ Following are statements made Friday and Thursday by Lawrence Wechsler, a lawyer for the White House secretary, Betty Currie; the White House; White House spokesman Mike McCurry, and President Clinton in response to an article in The New York Times on Friday about her statements regarding a meeting with the president: Wechsler on Thursday ``Without commenting on the allegations raised in this article, to the extent that there is any implication or suggestion that Mrs. Currie was aware of any legal or ethical impropriety by anyone, that implication or suggestion is entirely inaccurate.''"
    #sent = "Thursday's Markets: @ Earnings @ Data Cause @ Stock Fall @ --- @ Industrials Sink 39.55; @ Bonds Slip, but Dollar @ Soars Against Pound @ ---- @ By Douglas R. Sease @ Staff Reporter of The Wall Street Journal 10/27/89 WALL STREET JOURNAL (J) MONETARY NEWS, FOREIGN EXCHANGE, TRADE (MON) STOCK INDEXES (NDX) STOCK MARKET, OFFERINGS (STK) FINANCIAL, ACCOUNTING, LEASING (FIN) BOND MARKET NEWS (BON) FOREIGN-EXCHANGE MARKETS (FRX) TREASURY DEPARTMENT (TRE)"
    #start = 20
    #end = 20 + len(sent)
    start, end  = sent_span

    if re.search(r' \.+ ',sent):
        sentences = re.split(r' \.+ ', sent)
    if re.search(r', and, ',sent):
        sent = sent.replace(', and, ','. And, ')
        sentences = sent_tokenize(sent)
    if re.search(r'president\: Wechsler',sent):
        sent = sent.replace('president: Wechsler', 'president. Wechsler')
        sentences = sent_tokenize(sent)
    if re.search(r'@ ---- @', sent):
        sentences = re.split(r'@ ---- @', sent)

    sent_tokenize_span_list = spans(sentences, sent)
    sent_tokenize_span_list , max_len = add_start_end(sent_tokenize_span_list,start)
    #print sent_tokenize_span_list,max_len
    return sent_tokenize_span_list , max_len
