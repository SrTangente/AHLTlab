from os import listdir
import nltk
import re
from xml.dom.minidom import parse, parseString
from nltk.tokenize import word_tokenize
import evaluator


DATADIR = './data/devel/'
OUTFILENAME = './results.txt'


def load_drug_bank():
    """
    Load a the DrugBank file by creating a dictionary
    :return: the dictionary is in the form of {"drug_name": "drug|brand|group"}
    """
    drug_bank = {}
    with open('./resources/DrugBank.txt', 'r', encoding='utf8') as db:
        line = db.readline()
        while line:
            try:
                [name, type] = line.split('|')
                drug_bank[name.lower()] = type
                line = db.readline()
            except UnicodeDecodeError:
                print('Omitting line')

    return drug_bank


def tokenize(text):
    """
    :param text: the text to be split in tokens
    :return: a list of tokens triples, each one in the form of (word, start_offset, end_offset)
    """
    words = word_tokenize(text)
    return [(word, text.index(word), text.index(word)+len(word)-1) for word in words if "&" not in word]



def check_suffixes(words):
    prefixes = ['cef', 'ceph', 'cort', 'pred', 'sulfa']
    suffixes = ['azole', 'idine', 'amine', 'mycin', 'asone', 'bicin', 'afil', 'bital',
                'caine', 'cillin', 'cycline', 'dipine', 'dronate', 'eprazole', 'fenac',
                'floxacin', 'gliptin', 'glitazone', 'iramine', 'ine', 'mab', 'lamide', 'mustine',
                'mycin', 'nacin', 'olol', 'odone', 'olone', 'onide', 'parin', 'phylline', 'pril',
                'profen', 'ridone', 'sartan', 'semide', 'setron', 'statin', 'terol',
                'thiazide', 'tinib', 'trel', 'tretin', 'triptan', 'vir', 'vudine', 'zepam',
                'zolam', 'zosin']
    for pr in prefixes:
        for word in words:
            if word.startswith(pr):
                return True
    for suff in suffixes:
        for word in words:
            if word.endswith(suff):
                return True
    return False


def hasAlphaNum(word):
    return re.match("^(?=.*[A-Za-z])(?=.*[0-9])[A-Za-z0-9]*$", word) is not None

# process each file in directory
def extract_entities(tokens):
    drug_bank = load_drug_bank()
    entities = []
    for t in range(len(tokens)):
        word = tokens[t][0]
        i = 0
        words = word
        lower_words = word.lower()
        added = False
        while not added:
            entity = {
                'offset': f"{tokens[t][1]}-{tokens[t + i][2]}",
                'text': words
            }
            # Rules for classifying one single word
            if i == 0:
                if lower_words in drug_bank:
                    entity["type"] = drug_bank[lower_words]
                    entities.append(entity)
                    added = True
                '''elif check_suffixes([word]):
                    entity["type"] = "drug"
                    entities.append(entity)
                    added = True
                elif hasAlphaNum(word):
                    entity["type"] = "drug"
                    entities.append(entity)
                    added = True
                elif word.isupper():
                    entity["type"] = "brand"
                    entities.append(entity)
                    added = True'''

            # Rules for more than one word
            '''else:
                # Add the next token (word) to the entity
                try:
                    words = words + " " + tokens[t+i][0]
                    lower_words = words.lower()
                except IndexError:
                    continue

                if lower_words in drug_bank:
                    entity["type"] = drug_bank[lower_words]
                    entities.append(entity)
                    added = True'''

            i = i+1
            if i == 3 or i == len(tokens)-t:
                added = True
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
