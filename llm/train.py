import os, signal, time, sys
import math

from src.sampler import Sampler
from src.training_data import TrainingData
from src.devices import Devices
from src.llm import LLMSettings, LLM, LLMTraining
from src.llm_bigram import *


early_break = {
    'exit': False,
    'signal': signal.SIGUSR1
}


def receive_signal(signum, stack):
    if signum == early_break['signal']:
        early_break.update({'exit': True})

    print(f"Received signal {signum}, {'' if early_break['exit'] else 'not'} breaking", file=sys.stderr)


def in_steps(training: LLMTraining, tokenizer: Tokenizer, sampler: Sampler, llm_settings: LLMSettings, llm_model_read: str = None):

    # Machine
    device = Devices.getBestDevice()

    # Data
    data = TrainingData(device, tokenizer, sampler)

    # Model
    model = BigramLM(device, llm_settings, data).to(device)

    if llm_model_read:
        model.importModel(llm_model_read)


    # Start logging
    print(model, file=sys.stderr)
    print(tokenizer, file=sys.stderr)
    print(f"Running on {device}", file=sys.stderr)
    print(f"{model.model_parameters()} parameters", file=sys.stderr)
    print(f"To safely terminate the training use:", file=sys.stderr)
    print(f"   kill -n {early_break['signal']} {os.getpid()}", file=sys.stderr)

    # Prepare the early out
    early_break.update({'exit': False}) 
    signal.signal(early_break['signal'], receive_signal)

    # Train the model, parameters
    print(f"Training with {training}", file=sys.stderr)


    # Train the model, action
    optimizer = torch.optim.AdamW(model.parameters(), lr=llm_settings.learning_rate)
    time_start = time.time()
    time_last_report = time_start

    result = {}

    try:
        for steps in range(training.steps_total):

            if early_break['exit']:
                break

            report = False

            if training.report_timestep != None and time.time() - time_last_report > training.report_timestep:
                report = True
                time_last_report = time.time()

            if steps % training.steps_interval == 0:
                report = True

            if report:
                losses = model.estimate_loss(data, llm_settings, training.samples_to_calculate_loss)
                elapsed = math.floor(time.time() - time_start)
                eta = time.ctime(time_start + (elapsed * training.steps_total) / steps) if steps and training.report_timestep == None else '????'
                print(f"{elapsed}s {steps=} : {losses[LLM.TRAINING]=}  {losses[LLM.EVALUATION]=} eta={eta}", file=sys.stderr)

            model.trainStep(optimizer)

            result = { 'success': True, 'model': model, 'tokenizer': tokenizer }


    except KeyboardInterrupt:
        result = { 'success': False, 'model': model, 'tokenizer': tokenizer }


    return result
