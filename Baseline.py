from os import listdir
import nltk
from xml.dom.minidom import parse, parseString
from nltk.tokenize import word_tokenize
import evaluator

datadir = './data/train/'
outfile = './results.txt'

def tokenize(text):
    words = word_tokenize(text)
    return [(word, text.index(word), text.index(word)+len(word)) for word in words]

#process each file in directory
def extract_entities(tokens):
    db = open('DrugBank.txt', 'r').read()
    drugBank = word_tokenize(db)
    for t in tokens():
        word = t[0]
        if word in drugBank:
            return True, "drug"
        elif word.issupper(): return True, "brand"
        elif word[-5:] in ['azole', 'idine', 'amine', 'mycin']:
            return True, "drug"
        else:
            return False, ""


for f in listdir(datadir):
# parse XML file , obtaining a DOM tree
  tree = parse(datadir + "/" + f)
# process each sentence in the file
  sentences = tree.getElementsByTagName("sentence")
  for s in sentences :
    sid = s.attributes["id"].value # get sentence id
    stext = s.attributes["text"].value # get sentence text
    # tokenize text
    tokens = tokenize(stext)
    # extract entities from tokenized sentence text
    entities = extract_entities(tokens)
    # print sentence entities in format requested for evaluation
    for e in entities:
      print (sid +"|"+e["offset"]+"|"+ e["text"]+ "|"+ e["type"], file = outfile)
# print performance score
evaluator.evaluate("NER", datadir , outfile)# process each file in directory
