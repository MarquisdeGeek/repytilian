from os import listdir
from os import walk
from os.path import isfile, join

def load_textfile(filepath: str) -> str:
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    return text


def load_textfiles_from_path(dirpath: str) -> str:
    onlyfiles = [join(dirpath, f) for f in listdir(dirpath) if isfile(join(dirpath, f))]

    text = ''
    for filepath in onlyfiles:
        text += load_textfile(filepath)

    return text


def get_file_list(directoryName: str, extension: str) -> list:
    fnl = []
    for (dirpath, _, filenames) in walk(directoryName):
        [fnl.append(join(dirpath, f)) for f in filenames if f.endswith(extension)]

    return fnl

