import { Configuration, SharedFunctions } from './configuration';

/**
 * One neuron logic
 */
export class Neuron {
  public debug = false;
  private name = '';
  private inputValue = 0;
  private activatedValue = 0;

  public get output(): number {
    return this.activatedValue;
  }

  public set output(value: number) {
    this.activatedValue = value;
  }

  public set input(x: number) {
    this.inputValue = x + Configuration.bias;
    this.activatedValue = SharedFunctions.activationFunction(this.inputValue);
    this.log(
      `nr ${this.name} v: ${this.inputValue} -> ${this.activatedValue}, in: ${x}`
    );
  }

  public init(layerId: number, index: number): void {
    this.name = `${layerId}_${index}`;
  }

  public cost(expected: number): number {
    return SharedFunctions.costFunction(expected, this.activatedValue);
  }

  public prediction(weight: number, x: number): number {
    this.input = weight * x;
    return this.output;
  }

  private log = (logLine: string, args: string[] = []): void => {
    if (!this.debug) {
      return;
    }

    console.log(logLine, ...args);
  };
}
