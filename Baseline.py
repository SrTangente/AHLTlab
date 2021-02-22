from os import listdir
import nltk
from xml.dom.minidom import parse, parseString
from nltk.tokenize import word_tokenize
import evaluator


DATADIR = './data/train/'
OUTFILENAME = './results.txt'


def tokenize(text):
    words = word_tokenize(text)
    return [(word, text.index(word), text.index(word)+len(word)) for word in words]


# Process each file in directory
def extract_entities(tokens):
    db = open('./resources/DrugBank.txt', 'r').read()
    drugBank = word_tokenize(db)
    entities = []
    for t in tokens:
        word = t[0]
        entity = {
            'offset': f"{t[1]}-{t[2]}",
            'text': word
        }
        if word in drugBank:
            entity["type"] = "drug"
            entities.append(entity)
        elif word.isupper():
            entity["type"] = "brand"
            entities.append(entity)
        elif word[-5:] in ['azole', 'idine', 'amine', 'mycin']:
            entity["type"] = "drug"
            entities.append(entity)
    return entities


OUTFILE = open(OUTFILENAME, 'w+')

for f in listdir(DATADIR):
    # parse XML file , obtaining a DOM tree
    tree = parse(DATADIR + "/" + f)
    # process each sentence in the file
    sentences = tree.getElementsByTagName("sentence")
    for s in sentences:
        sid = s.attributes["id"].value  # get sentence id
        stext = s.attributes["text"].value  # get sentence text
        # tokenize text
        tokens = tokenize(stext)
        # extract entities from tokenized sentence text
        entities = extract_entities(tokens)
        # print sentence entities in format requested for evaluation
        for e in entities:
            print(sid +"|" + e["offset"] +"|" + e["text"] + "|" + e["type"], file=OUTFILE)

# print performance score
evaluator.evaluate("NER", DATADIR, OUTFILE)# process each file in directory
