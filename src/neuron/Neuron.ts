import { Configuration, SharedFunctions } from './configuration';

/**
 * One neuron logic
 */
export class Neuron {
  public neuronId = 0;
  public debug = false;
  public propagationError = 0;

  private name = '';
  private activatedValue = 0;
  private propagationSum = 0;
  private weights: number[] = [];

  public get output(): number {
    return this.activatedValue;
  }

  public set output(value: number) {
    this.activatedValue = value;
  }

  private set input(value: number) {
    this.activatedValue = SharedFunctions.activationFunction(value);
    this.log(
      `nr ${this.name} v: ${value} -> ${this.activatedValue}, in: ${value}`
    );
  }

  public init(layerId: number, neuronId: number): void {
    this.neuronId = neuronId;
    this.name = `${layerId}_${neuronId}`;
  }

  public cost(expected: number): number {
    return SharedFunctions.costFunction(expected, this.activatedValue);
  }

  public propagate(linkIndex: number, linkValue: number): number {
    let weight = this.weights[linkIndex];
    if (weight === undefined) {
      weight = SharedFunctions.initialWeight(); // TODO: in order to optimize this we could prefill weights oat the init step
      this.weights[linkIndex] = weight;
    }

    this.propagationSum += weight * linkValue;
    return this.output;
  }

  /**
   * Makes neuron prediction according to signals on inputs
   */
  public prediction(): void {
    // this.state = NeuronState.Prediction;
    this.input = this.propagationSum;
    this.propagationSum = Configuration.bias;
  }

  public weightError = (inputId: number): number => {
    return this.weights[inputId] * this.propagationError;
  };

  correctWeights = (learningDelta: number): void => {
    for (let i = 0; i < this.weights.length; i++) {
      const newWeight =
        this.weights[i] +
        this.propagationError *
          SharedFunctions.activationFunctionPrime(this.output) *
          this.propagationSum *
          learningDelta;
      this.weights[i] = newWeight;
    }
  };

  private log = (logLine: string, args: string[] = []): void => {
    if (!this.debug) {
      return;
    }

    console.log(logLine, ...args);
  };
}
