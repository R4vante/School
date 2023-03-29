from neural import Neural
import sys

def main():
    neural = Neural("mnist", sys.argv[2])
    
    model_small, results_small, y_pred_small = neural.model(sys.argv[1])

    neural.plot()


if __name__ == "__main__":
    main()
