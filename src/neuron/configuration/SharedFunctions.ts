import { Configuration } from './Configuration';

/**
 * Shared functions
 */
class SharedFunctions {
  /**
   * Initial random weight interval is probably depends on elcted activation function
   * TODO: !check that
   */
  public initialWeight = (): number => {
    switch (Configuration.activationType) {
      case 'ReLU':
        return Math.random(); // [0, 1]
      case 'LeakyReLU':
        return Math.random(); // [0, 1]
      case 'Sigmoid':
        return Math.random() - 1; // [-0.5, 0.5]
      default: {
        console.warn(
          `Define initial weight function for ${Configuration.activationType} actiovation type`
        );
        return Math.random(); // [0, 1]
      }
    }
  };

  // more https://rohanvarma.me/Loss-Functions/
  // more https://github.com/trekhleb/nano-neuron
  public costFunction = (expected: number, prediction: number): number => {
    switch (Configuration.useCostFunction) {
      case 'Squared': {
        // every time returns a result in interval [0, +inf)
        // usually used for regression
        const v =
          expected * expected -
          2 * expected * prediction +
          prediction * prediction;
        return v * 0.5;
      }
      // case 'CrossEntropy':  // TODO: needs for classification
      default:
        return expected - prediction;
    }
  };

  public activationFunction = (x: number): number => {
    switch (Configuration.activationType) {
      case 'ReLU':
        return this.activationFunctionReLU(x);
      case 'LeakyReLU':
        return this.activationFunctionLeakyReLU(x);
      case 'Sigmoid':
        return this.activationFunctionSigmoid(x);
      default:
        return this.activationFunctionIdentity(x);
    }
  };

  public activationFunctionPrime = (x: number): number => {
    switch (Configuration.activationType) {
      case 'ReLU':
        return this.activationFunctionReLUPrime(x);
      case 'LeakyReLU':
        return this.activationFunctionLeakyReLUPrime(x);
      case 'Sigmoid':
        return this.activationFunctionSigmoidPrime(x);
      default:
        return this.activationFunctionIdentityPrime(x);
    }
  };

  // more: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
  private activationFunctionReLU = (x: number): number => {
    return x > 0 ? x : 0;
  };

  /**
   * Streats slope as 45 degress when x > 0 and 0 degress for all other cases
   */
  private activationFunctionReLUPrime = (x: number): number => {
    return x > 0 ? 1 : 0;
  };

  private activationFunctionLeakyReLU = (x: number): number => {
    return x > 0 ? x : 0.01 * x;
  };

  private activationFunctionLeakyReLUPrime = (x: number): number => {
    return x > 0 ? 1 : 0.01;
  };

  // more: https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e
  private activationFunctionSigmoid = (x: number): number => {
    return 1 / (1 + Math.pow(Math.E, -x));
  };

  private activationFunctionSigmoidPrime = (x: number): number => {
    return (
      this.activationFunctionSigmoid(x) *
      (1 - this.activationFunctionSigmoid(x))
    );
  };

  private activationFunctionIdentity = (x: number): number => {
    return x;
  };

  // eslint-disable-next-line @typescript-eslint/no-unused-vars, no-unused-vars
  private activationFunctionIdentityPrime = (_x: number): number => {
    return 1;
  };
}

const sharedFunctions = new SharedFunctions();
export { sharedFunctions as SharedFunctions };
