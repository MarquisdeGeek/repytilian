import src.tokenizer as tokenizer
from src.devices import Devices
from src.llm import *
from src.llm_bigram import *


def create_model(llm_model):

  # Machine
  device = Devices.getBestDevice()

  # Data
  tokenizer_inst = tokenizer.importTokens(llm_model)

  # Model
  settings = LLMSettings()
  model = BigramLM(device, settings, tokenizer_inst).to(device)
  model.importModel(llm_model)

  return model


def new_output(llm_model, token_count):

  model = create_model(llm_model)

  # Do the work
  output = model.generateOutput(token_count)

  # Provide it
  return { 'output': output, 'model': model }


def continue_output(llm_model, token_count, initial_value):

  model = create_model(llm_model)

  # Do the work
  output = model.generateOutputFrom(initial_value, token_count)

  # Provide it
  return { 'output': output, 'model': model }

