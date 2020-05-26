/**
 * Main settings fo the ML system
 */
class Configuration {
  public bias = 1;
  /** Activation function */
  public activationType = 'Identity';
  /** Cost function */
  public useCostFunction = 'Squared';

  // more https://rohanvarma.me/Loss-Functions/
  // more https://github.com/trekhleb/nano-neuron
  public costFunction = (expected: number, prediction: number): number => {
    switch (this.useCostFunction) {
      case 'Squared': {
        const v = expected * expected - 2 * expected * prediction + prediction * prediction;
        return v * 0.5;
      }
      default:
        return expected - prediction;
    }
  };

  public activationFunction = (x: number): number => {
    switch (this.activationType) {
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
    switch (this.activationType) {
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
    return this.activationFunctionSigmoid(x) * (1 - this.activationFunctionSigmoid(x));
  };

  private activationFunctionIdentity = (x: number): number => {
    return x;
  };

  // eslint-disable-next-line @typescript-eslint/no-unused-vars, no-unused-vars
  private activationFunctionIdentityPrime = (_x: number): number => {
    return 1;
  };
}

const configuration = new Configuration();
export { configuration as Configuration };