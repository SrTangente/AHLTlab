from os import listdir
import nltk
from xml.dom.minidom import parse, parseString
from nltk.tokenize import word_tokenize
import evaluator


DATADIR = './data/train/'
OUTFILENAME = './results.txt'


def load_drug_bank():
    """
    Load a the DrugBank file by creating a dictionary
    :return: the dictionary is in the form of {"drug_name": "drug|brand|group"}
    """
    drug_bank = {}
    with open('./resources/DrugBank.txt', 'r') as db:
        line = db.readline()
        while line:
            [name, type] = line.split('|')
            drug_bank[name] = type
            line = db.readline()

    return drug_bank


def tokenize(text):
    """
    :param text: the text to be split in tokens
    :return: a list of tokens triples, each one in the form of (word, start_offset, end_offset)
    """
    words = word_tokenize(text)
    return [(word, text.index(word), text.index(word)+len(word)-1) for word in words if "&" not in word]


# process each file in directory
def extract_entities(tokens):
    drug_bank = load_drug_bank()
    entities = []
    for t in tokens:
        word = t[0]
        entity = {
            'offset': f"{t[1]}-{t[2]}",
            'text': word
        }
        if word in drug_bank:
            entity["type"] = drug_bank[word]
            entities.append(entity)
        elif word.isupper():
            entity["type"] = "brand"
            entities.append(entity)
        elif word[-5:] in ['azole', 'idine', 'amine', 'mycin']:
            entity["type"] = "drug"
            entities.append(entity)
    return entities


with open(OUTFILENAME, 'w') as OUTFILE:
    for f in listdir(DATADIR):
        # parse XML file , obtaining a DOM tree
        tree = parse(DATADIR + "/" + f)
        # process each sentence in the file
        sentences = tree.getElementsByTagName("sentence")
        for s in sentences:
            sid = s.attributes["id"].value  # get sentence id
            stext = s.attributes["text"].value  # get sentence text
            # tokenize text
            try:
                tokens = tokenize(stext)
            except ValueError:
                print(f"Omitting '{stext}' because a ValueError")
                continue
            # extract entities from tokenized sentence text
            entities = extract_entities(tokens)
            # print sentence entities in format requested for evaluation
            for e in entities:
                print(sid +"|" + e["offset"] +"|" + e["text"] + "|" + e["type"], file=OUTFILE)

# print performance score
evaluator.evaluate("NER", DATADIR, OUTFILENAME)# process each file in directory
