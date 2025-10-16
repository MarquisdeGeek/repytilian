
# About

Repytilian is an educational project, demonstrating how LLMs can be trained, and text generated from that model. There are two primary entry points, one for training a model from a directory of text files, and a second for generating data from that pre-computed/pre-saved model.

It is based on the code from Andre's video detailed below.


Disclaimer: This code is an OO variation of the code from the video. It contains a few improvements. It exists primarily to keep my hand in with some AI and Python development.


# Installation

Just do:

```
pip3 install -r requirements.txt
```

Or, for a less verbose set of modules:
```
  pip install torch numpy transformers datasets tiktoken wandb tqdm
```


# Examples

A quick and basic training on the Shakespeare dataset:
```
python3 training.py
```

This generates a model in the directory `models/shakespeare/basic` which you can later re-use with:

```
python3 generate.py
```
By adding the `-m` switch, you can use one of the pre-trained models:
```
python3 generate.py -m models/shakespeare/v2c
```

To invoke an auto-complete, give the model some valid tokens to start:
```
python3 generate.py -m models/shakespeare/v2c -c "My lady"
```
Note: if the model doesn't not contain those tokens (as can happen on word-based tokenizers) this will fail.

# Working with other models

Training is our first step in creating an LLM, from prepared data in plain ASCII. To do this you must decide on:

1. The source data for the model
1. How the text is to be converted into tokens
1. For how long the training will last

The Source data is simply a text file, of set of files in a directory. The training script will read all the text in the given directory for processing. For an individual file specify it with `-d`, while and entire directory is determined with the `-p` flag.

The second case is known as tokenisation, and uses one of the methods found in `src/tokenizers`. Usually you want `characters` or `words`. Specify which with the `-z` parameter.

Finally, the duration is measured in iterations. To minimize the number of parameters, we have a simple set of defaults for this. So specify either  `default`, `quick`, `long`, `forever`
with the `-i` parameter.

So when using `-i` or perpetual training, it will keep training until you want your CPU to do something more interesting! In which case hit Ctrl+C to stop. It will save the current state in the `interrupted` folder, which you can then move to a more permanent directory if necessary. We don't overwrite the model file in this case, in case you start the training accidentally. Instead of Crl+C, you can also end the training by sending a signal such as `kill -n 10 1722845`, where the number is the process ID and presented on-screen when training begins.

To experiment with training, see a sample, but not save anything:
```
python3 generate.py -m ""
```


# Other examples


## Shakespeare

There are three models, all character based, each one with more training than the one that preceded it. So do compare:
```
python3 generate.py -m models/shakespeare/v1c
```
with
```
python3 generate.py -m models/shakespeare/v2c
```
and
```
python3 generate.py -m models/shakespeare/v3c
```



## Star Wars
Write your own, 1000 word, Star Wars script with:

```
python3 generate.py -m models/starwars/v1w -t 1000
```
Note that if you use the continuation function, the initial words must exist in the dataset because the scripts were tokenized with words. So:
```
python3 generate.py -m models/starwars/v1w -c Luke
```
works, while:
```
python3 generate.py -m models/starwars/v1w -c luke
```
fails.

For a model which was tokenized with characters you will get an answer with either input... but the output itself may be less than optimal!

```
python3 generate.py -m models/starwars/v0c -c luke
```


## Me
I even trained a model with my books. The output is not representative of my work!
```
python3 generate.py -m models/steev/v2w
```

Similarly, there is a character-based token version for comparison. See the results with:
```
python3 generate.py -m models/steev/v0c
```

# Basic tool usage

The (current) command line options are:

## Training
```
$ python3 training.py --help
usage: training.py [-h] [-d DATAFILE] [-p DATAPATH] [-l LOAD] [-m MODEL] [-s SETTINGS] [-i ITERATIONS] [-t TOKENS] [-z TOKENIZER]

Train text with a bigram model

options:
  -h, --help            show this help message and exit
  -d DATAFILE, --datafile DATAFILE
                        Single filepath used for training data
  -p DATAPATH, --datapath DATAPATH
                        Directory where all files within are used for training data
  -l LOAD, --load LOAD  Start from a previously-exported model
  -m MODEL, --model MODEL
                        Target folder for model export. Defaults to shakespeare. Use "" to save nothing
  -s SETTINGS, --settings SETTINGS
                        Config for training. (e.g. default,hyper)
  -i ITERATIONS, --iterations ITERATIONS
                        Iteration optios used for training. (e.g. default,quick,long,forever)
  -t TOKENS, --tokens TOKENS
                        Number of tokens to generate
  -z TOKENIZER, --tokenizer TOKENIZER
                        Type of tokenizer to use (e.g. characters,words)
```

## Generation
```
$ python3 generate.py --help
usage: generate.py [-h] [-m MODEL] [-t TOKENS] [-c CONTINUE] [-f FILENAME] [-ft FILETOKENS]

Build text from a bigram model

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Filename of previously-exported target model
  -t TOKENS, --tokens TOKENS
                        Number of tokens to generate
  -c CONTINUE, --continue CONTINUE
                        Continue generation from the given text
  -f FILENAME, --filename FILENAME
                        Write generated tokens to file, as well/instead of stdout
  -ft FILETOKENS, --filetokens FILETOKENS
                        Number of tokens to write into file, if different
```

# Additions

Some of the things not discussed in the video:

* A word tokenizer
* Arbitrary JSON tokenizer. Currently used for music LLMs (WIP)
* Ability to load and save models
* Training continuation (so you can pause generation, save the state, and reload later)
* Control+C (aka SIGINT) will save a temp model (which you can later continue)
* Argument parsing, to permit alternate data sets and models
* OO code with type annotations, classes, and other Pythonic stuff




# Running tests

Such that they are:

```
python3 -m unittest
```


# Learning materials

https://www.youtube.com/watch?v=kCc8FmEb1nY&ab_channel=AndrejKarpathy

https://github.com/karpathy/ng-video-lecture
