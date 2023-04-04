from neural import Neural
import sys

def main():
    
    # Declaring an object for initialization
    neural = Neural("mnist", sys.argv[2], sys.argv[1])
    
    # Training model with results and prediction
    model_small, results_small, y_pred_small = neural.model()

    # plot the results of the training as well as the first predictions of testing
    neural.plot()


if __name__ == "__main__":
    main()
