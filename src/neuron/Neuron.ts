import { Log } from '../services';
import { Configuration, SharedFunctions } from './configuration';
import { StringFunctions } from '.';
import { ActivationType } from './models';

// shortcut to rounding function
// eslint-disable-next-line no-unused-vars, @typescript-eslint/no-unused-vars
const fnz = StringFunctions.fnz;

/**
 * One neuron logic
 */
export class Neuron {
  public neuronId = 0;
  public layerId = 0;
  /** propagation error, difference between current output and expected  output */
  public errorSignal = 0;

  private moduleName = '';
  private activatedValue = 0;
  private propagationSum = Configuration.bias;
  private inputs: number[] = [];
  private weights: number[] = [];

  public get output(): number {
    return this.activatedValue;
  }

  public set output(value: number) {
    this.activatedValue = value;
  }

  private set input(value: number) {
    this.activatedValue = SharedFunctions.activationFunction(
      value,
      this.activationType
    );
    Log.debug(
      `Out: act(${fnz(value)}) -> ${fnz(this.activatedValue)}`,
      this.moduleName
    );
  }

  constructor(
    layerId: number,
    neuronId: number,
    private activationType: ActivationType
  ) {
    this.neuronId = neuronId;
    this.layerId = layerId;
    this.moduleName = `Nr ${neuronId + 1}${layerId}`;
    Log.debug(`Config: activation: ${this.activationType}`, this.moduleName);
  }

  /** Find propagation error, i.e..difference between current output and expected  output */
  public findErrorSignal(expected: number): number {
    //this.log(`costf expec: ${fnz(expected)}, act: ${fnz(this.output)}`);
    Log.debug(
      `error signal:`,
      this.moduleName,
      this.output,
      `expeced:`,
      expected
    );
    const errorCost = SharedFunctions.errorSignalFunction(
      expected,
      this.output
    );
    return errorCost;
  }

  /**
   * Propagate method
   * @param inputId Link index
   * @param linkValue value
   */
  public propagate(inputId: number, linkValue: number): void {
    let weight = this.weights[inputId];
    if (weight === undefined) {
      weight = SharedFunctions.initialWeight(); // TODO: in order to optimize this we could prefill weights oat the init step
      this.weights[inputId] = weight;
    }

    // Log.debug(`${this.wIndex(inputId)} = ${fnz(weight)}`);
    this.inputs[inputId] = linkValue;

    //const pSum = this.propagationSum;
    this.propagationSum += weight * linkValue;
    // Log.debug(
    //   `Sum: ${fnz(pSum)} -> ${fnz(this.propagationSum)} `,
    //   fnz(weight),
    //   '*',
    //   fnz(linkValue)
    // );
  }

  /**
   * Makes neuron prediction according to signals on inputs
   */
  public prediction(): void {
    // this.state = NeuronState.Prediction;
    this.input = this.propagationSum;
    this.propagationSum = Configuration.bias;

    Log.debug(`prediction ${fnz(this.output)}`, this.moduleName);
  }

  /**
   * Returns  weight(input) * Error sygnal
   * i.e. part of error according to weigh on given input
   */
  public weightedErrorSignal = (inputId: number): number => {
    const errorByWeight = this.weights[inputId] * this.errorSignal;
    Log.debug(
      `weightedErrorSignal ${this.wIndex(inputId)} = ${fnz(errorByWeight)}`,
      this.moduleName,
      this.weights[inputId],
      this.errorSignal
    );
    return errorByWeight;
  };

  public correctWeights = (learningDelta: number): void => {
    const weightAdditionMultiplayer =
      this.errorSignal *
      SharedFunctions.activationFunctionPrime(
        this.output,
        this.activationType
      ) *
      learningDelta;

    if (!isFinite(weightAdditionMultiplayer)) {
      Log.throw(
        `${this.neuronId}${this.layerId} addition Infinity, propError ${this.errorSignal}, my out ${this.output}`,
        this.moduleName,
        weightAdditionMultiplayer
      );
      return;
    }

    for (let i = 0; i < this.weights.length; i++) {
      const weightAddition = weightAdditionMultiplayer * this.inputs[i];
      this.weights[i] += weightAddition;

      if (isNaN(this.weights[i])) {
        Log.throw(
          `${this.wIndex(i)} input is NAN, weightAddition ${weightAddition}`,
          this.moduleName,
          weightAdditionMultiplayer,
          this.inputs[i]
        );
        return;
      }

      // this.log(`weightError ${this.wIndex(i)} = ${fnz(this.weights[i])}, added ${fnz(weightAddition)}`);
    }
  };

  /** Debug method. Allows to set weights directly */
  public initWeights = (weights: number[]): void => {
    // this.log('Ne', weights);
    this.weights = weights;
  };

  /** Debug method. Allows to set weights directly */
  public getWeights = (): number[] => {
    // this.log('GNe', weights);
    return this.weights;
  };

  private wIndex = (inputId: number): string =>
    `W${inputId + 1}${this.neuronId}${this.layerId}`;
}
