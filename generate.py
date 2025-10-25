import sys
import argparse
from llm import generate


# Init
parser = argparse.ArgumentParser(description='Build text from a bigram model')
parser.add_argument('-m', '--model', help='Filename of previously-exported target model')
parser.add_argument('-t', '--tokens', help='Number of tokens to generate')
parser.add_argument('-c', '--continue', help='Continue generation from the given text')
parser.add_argument('-f', '--filename', help='Write generated tokens to file, as well/instead of stdout')
parser.add_argument('-ft', '--filetokens', help='Number of tokens to write into file, if different')
args = parser.parse_args()

# Prep globals
llm_model = 'models/shakespeare/basic' if args.model == None else args.model
value_initial = getattr(args, 'continue')
token_stdout_count = 100 if args.tokens == None else int(args.tokens)
token_file_count = token_stdout_count if args.filetokens == None else int(args.filetokens)
token_filename = args.filename


# Get the model
model = generate.create_model(llm_model)
print(model, file=sys.stderr)
print(model.tokenizer_inst, file=sys.stderr)


# 1. Generate text, starting from 'nothing'
# (in reality, assumes an entry point of the first token, current NL)
if not value_initial:
  print(f"LLM generating output ({token_stdout_count} tokens):", file=sys.stderr)
  print(model.generate_output(token_stdout_count))
  # Or
  # print(generate.new_output(llm_model, token_stdout_count)['output'])


# 2. Auto-complete, from the given input
if value_initial:
  print(f"LLM continuing output from '{value_initial}' ({token_stdout_count} tokens):", file=sys.stderr)
  print(model.generate_output_from(value_initial, token_stdout_count))
  # Or
  # print(generate.continue_output("models/steev/v0w", 100,"Because")['output'])

# 3. File output
if token_filename:
  print(f"LLM generating file '{token_filename}' ({token_file_count} tokens):", file=sys.stderr)
  open(token_filename, 'w').write(model.generate_output(token_file_count))

