from os import listdir
import nltk
from xml.dom.minidom import parse, parseString
from nltk.tokenize import word_tokenize
import evaluator
from nltk.corpus import stopwords

DATADIR = './data/train/'
OUTFILENAME = './results.txt'


def tokenize(text):
    words = word_tokenize(text)
    print(words)
    result = []
    for word in words:
        try:
            result.append((word, text.index(word), text.index(word)+len(word)))
        except:
            print("Some word could not be readed")
    return result

def get_resources():
    db = open('./resources/DrugBank.txt', 'r', encoding="latin1").read()
    drugBank = set(word_tokenize(db))
    punctuation = [',', '.', ':', ';', '?', '!', '"', "'"]
    sw = set(stopwords.words('english'))
    for w in sw:
        if w in drugBank:
            drugBank.remove(w)
    for w in punctuation:
        if w in drugBank:
            drugBank.remove(w)
    return [drugBank]

# Process each file in directory
def extract_entities(tokens, resources):
    drugBank = resources[0]
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

line = 0
OUTFILE = open(OUTFILENAME, 'w+')

resources = get_resources()
for f in listdir(DATADIR):
    line = line + 1
    print(line)
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
        entities = extract_entities(tokens, resources)
        # print sentence entities in format requested for evaluation
        for e in entities:
            print(sid +"|" + e["offset"] +"|" + e["text"] + "|" + e["type"], file=OUTFILE)

# print performance score
evaluator.evaluate("NER", DATADIR, OUTFILE)
# process each file in directory
