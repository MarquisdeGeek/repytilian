import sys
import argparse
import time
import math
import torch

from llm import train


import src.importer as importer
import src.tokenizer as tokenizer
from src.samplers.truncate import BasicTruncate
from src.training_data import TrainingData
from src.devices import Devices
from src.llm import LLMSettings, LLM, LLMTraining
from src.llm_bigram import *


# Init
parser = argparse.ArgumentParser(description='Train text with a bigram model')
parser.add_argument('-d', '--datafile', help='Single filepath used for training data')
parser.add_argument('-p', '--datapath', help='Directory where all files within are used for training data')
parser.add_argument('-l', '--load', help='Start from a previously-exported model')
parser.add_argument('-m', '--model', help='Target folder for model export. Defaults to shakespeare. Use "" to save nothing')
parser.add_argument('-s', '--settings', help=f'Config for training. (e.g. {",".join(LLMSettings.OPTIONS)})')
parser.add_argument('-i', '--iterations', help=f'Iteration optios used for training. (e.g. {",".join(LLMTraining.OPTIONS)})')
parser.add_argument('-t', '--tokens', help='Number of tokens to generate')
parser.add_argument('-z', '--tokenizer', help=f'Type of tokenizer to use (e.g. {",".join(Tokenizer.OPTIONS)})')
args = parser.parse_args()


# Prep globals from options
data_file = 'datasets/shakespeare/1m' if  args.datafile == None else args.datafile
data_path = args.datapath
tokenizer_type = args.tokenizer if args.tokenizer else Tokenizer.DEFAULT
training_iterations = args.iterations if args.iterations else LLMTraining.DEFAULT
llm_model_settings = args.settings if args.settings else LLMSettings.DEFAULT
llm_model_read = args.load
llm_model_write = 'models/shakespeare/basic' if args.model == None else args.model
token_stdout_count = 100 if args.tokens == None else int(args.tokens)


# Validate options
if not tokenizer_type in Tokenizer.OPTIONS:
  print(f'Err: "{tokenizer_type}" not understood as a tokenizer option ({",".join(Tokenizer.OPTIONS)})', file=sys.stderr)
  exit(-1)

if not training_iterations in LLMTraining.OPTIONS:
  print(f'Err: "{llm_model_settings}" not understood as a training option ({",".join(LLMTraining.OPTIONS)})', file=sys.stderr)
  exit(-2)

if not llm_model_settings in LLMSettings.OPTIONS:
  print(f'Err: "{llm_model_settings}" not understood as a settings option ({",".join(LLMSettings.OPTIONS)})', file=sys.stderr)
  exit(-3) 


# Train the model, parameters
training = LLMTraining(training_iterations)
training.steps_interval = training.steps_total // training.steps_logging
print(f"Training with {training}")


# Data
if data_path:
  text = importer.loadTextFilesFromPath(data_path)
else:
  text = importer.loadTextFile(data_file)


# Model
llm_settings = LLMSettings(llm_model_settings)
sampler = BasicTruncate(0.9)
tokenizer = tokenizer.tokenize(tokenizer_type, text)


# Do it
results = train.in_steps(training, tokenizer, sampler, llm_settings, llm_model_read)


# Save the model?
if results['success']:
  if llm_model_write:
      results['model'].exportModel(llm_model_write)
      results['tokenizer'].exportTokens(llm_model_write)
else:
    filepath = f'models/interrupted/at_{int(time.time())}'
    print(f"LLM training interrupted. Saving temp model to {filepath}", file=sys.stderr)
    results['model'].exportModel(filepath)
    results['tokenizer'].exportTokens(filepath)


# Give an example output
print(results['model'].generateOutput(token_stdout_count))

exit(0)
#// TODO: REMOVE exit, and junk below

# Machine
device = Devices.getBestDevice()


# Data
if data_path:
  text = importer.loadTextFilesFromPath(data_path)
else:
  text = importer.loadTextFile(data_file)

tokenizer = tokenizer.tokenize(tokenizer_type, text)


data = TrainingData(device, tokenizer, sampler)


# Model
settings = LLMSettings(llm_model_settings)
model = BigramLM(device, settings, data).to(device)

if llm_model_read:
  model.importModel(llm_model_read)


# Start logging
print(model)
print(tokenizer)
print(f"Running on {device}")
print(f"{model.model_parameters()} parameters")


# Train the model, parameters
training = LLMTraining(training_iterations)
training.steps_interval = training.steps_total // training.steps_logging
print(f"Training with {training}")


# Train the model, action
optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate)
time_start = time.time()
time_last_report = time_start

try:
  for steps in range(training.steps_total):

    report = False

    if training.report_timestep != None and time.time() - time_last_report > training.report_timestep:
      report = True
      time_last_report = time.time()

    if steps % training.steps_interval == 0:
      report = True

    if report:
      losses = model.estimate_loss(data, settings, training.samples_to_calculate_loss)
      elapsed = math.floor(time.time() - time_start)
      eta = time.ctime(time_start + (elapsed * training.steps_total) / steps) if steps and training.report_timestep == None else '????'
      print(f"{elapsed}s {steps=} : {losses[LLM.TRAINING]=}  {losses[LLM.EVALUATION]=} eta={eta}")

    model.trainStep(optimizer)

    if llm_model_write:
      model.exportModel(llm_model_write)
      tokenizer.exportTokens(llm_model_write)

except KeyboardInterrupt:
  filepath = f'models/interrupted/at_{int(time.time())}'
  print(f"LLM training interrupted. Saving temp model to {filepath}", file=sys.stderr)
  model.exportModel(filepath)
  tokenizer.exportTokens(filepath)


# Give an example output
print(model.generateOutput(token_stdout_count))

