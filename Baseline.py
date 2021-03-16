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
                drug_bank[name.lower()] = type.rstrip()
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
    # TODO: change the way we get the offset for the token, since with "index" a word that appears more than once
    #  in a sentence is assigned always the offset of its first appearance
    return [(word, text.index(word), text.index(word)+len(word)-1) for word in words if "&" not in word]


def check_affixes(words):
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
    t = 0 # the start of the tokens we are currently looking
    while t < len(tokens):
        word = tokens[t][0]
        entity = {
            'offset': f"{tokens[t][1]}-{tokens[t][2]}",
            'text': word
        }

        # See if token alone, or combined with the following, exists in the external resources
        end = 0 # relative position of 't' of the ending token
        words = word
        lower_words = word.lower()
        added = False
        while added == False and end < 4 and end < len(tokens)-t:

            if lower_words in drug_bank:
                entity["type"] = drug_bank[lower_words]
                entities.append(entity)
                added = True

            # Add the next token (word) to the entity
            try:
                end = end + 1
                words = words + " " + tokens[t + end][0]
                lower_words = words.lower()
                entity = {
                    'offset': f"{tokens[t][1]}-{tokens[t + end][2]}",
                    'text': words
                }
            except IndexError:
                continue

        # If the token (maybe combined with the following) existed in external resources,
        # just continue with the next iteration
        if added:
            t = t + end + 1
            continue

        # Otherwise, check rules for classifying one single word
        if check_affixes([word]):
            entity["type"] = "drug"
            entities.append(entity)
        elif hasAlphaNum(word):
            entity["type"] = "drug"
            entities.append(entity)
        elif word.isupper():
            entity["type"] = "brand"
            entities.append(entity)

        # Update position for next iteration
        t = t + 1
    return entities


if __name__ == "__main__":

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
