from game import loop_game
import argparse
from brain import Brain
import torch
import pdb

def main():
    parser = argparse.ArgumentParser(description='Evaluate a PyTorch model.')
    parser.add_argument('--model', required=False, help='Path to the PyTorch model file')

    args = parser.parse_args()

    #TODO: Verify the args path is a valid model

    if args.model:
        model = Brain()
        #load torch model located at path
        if args.model != "new":
            model.load_state_dict(torch.load(args.model), strict=True)
    else:
        model = None    

    loop_game(model, virtual=False, random_seed=0)



if __name__ == "__main__":
    main()




