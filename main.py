from game import loop_game
from nn import evaluate_network
import argparse
from brain import Brain
import torch

def main():
    parser = argparse.ArgumentParser(description='Evaluate a PyTorch model.')
    parser.add_argument('--model', required=True, help='Path to the PyTorch model file')

    args = parser.parse_args()

    #TODO: Verify the args path is a valid model

    if args.model:
        model = Brain()
        #load torch model located at path
        model = model.load_state_dict(torch.load(args.model))
    else:
        model = None    

    loop_game(model)



if __name__ == "__main__":
    main()




