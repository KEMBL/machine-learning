import { Configuration } from './Configuration';

/**
 * Shared functions
 */
export class SharedFunctions {
  /**
   * Initial random weight interval is probably depends on elcted activation function
   * TODO: !check that
   */
  public static initialWeight = (): number => {
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
  public static costFunction = (
    expected: number,
    prediction: number
  ): number => {
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
        // Identity
        return expected - prediction;
    }
  };

  public static activationFunction = (x: number): number => {
    switch (Configuration.activationType) {
      case 'ReLU':
        return SharedFunctions.activationFunctionReLU(x);
      case 'LeakyReLU':
        return SharedFunctions.activationFunctionLeakyReLU(x);
      case 'Sigmoid':
        return SharedFunctions.activationFunctionSigmoid(x);
      default:
        // Identity
        return SharedFunctions.activationFunctionIdentity(x);
    }
  };

  public static activationFunctionPrime = (x: number): number => {
    switch (Configuration.activationType) {
      case 'ReLU':
        return SharedFunctions.activationFunctionReLUPrime(x);
      case 'LeakyReLU':
        return SharedFunctions.activationFunctionLeakyReLUPrime(x);
      case 'Sigmoid':
        return SharedFunctions.activationFunctionSigmoidPrime(x);
      default:
        // Identity
        return SharedFunctions.activationFunctionIdentityPrime(x);
    }
  };

  // more: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
  private static activationFunctionReLU = (x: number): number => {
    return x > 0 ? x : 0;
  };

  /**
   * Streats slope as 45 degress when x > 0 and 0 degress for all other cases
   */
  private static activationFunctionReLUPrime = (x: number): number => {
    return x > 0 ? 1 : 0;
  };

  private static activationFunctionLeakyReLU = (x: number): number => {
    return x > 0 ? x : 0.01 * x;
  };

  private static activationFunctionLeakyReLUPrime = (x: number): number => {
    return x > 0 ? 1 : 0.01;
  };

  // more: https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e
  private static activationFunctionSigmoid = (x: number): number => {
    return 1 / (1 + Math.pow(Math.E, -x));
  };

  private static activationFunctionSigmoidPrime = (x: number): number => {
    return (
      SharedFunctions.activationFunctionSigmoid(x) *
      (1 - SharedFunctions.activationFunctionSigmoid(x))
    );
  };

  private static activationFunctionIdentity = (x: number): number => {
    return x;
  };

  // eslint-disable-next-line @typescript-eslint/no-unused-vars, no-unused-vars
  private static activationFunctionIdentityPrime = (_x: number): number => {
    return 1;
  };
}
