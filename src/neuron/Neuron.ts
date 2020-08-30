import { Configuration, SharedFunctions } from './configuration';

/**
 * One neuron logic
 */
export class Neuron {
  public debug = false;
  private name = '';
  private inputValue = 0;
  private activatedValue = 0;
  private _weight = 0;

  public get output(): number {
    return this.activatedValue;
  }

  public set output(value: number) {
    this.activatedValue = value;
  }

  public get weight(): number {
    return this._weight;
  }

  public set weight(value: number) {
    this._weight = value;
  }

  private set input(x: number) {
    this.inputValue = x + Configuration.bias;
    this.activatedValue = SharedFunctions.activationFunction(this.inputValue);
    this.log(
      `nr ${this.name} v: ${this.inputValue} -> ${this.activatedValue}, in: ${x}`
    );
  }

  public init(layerId: number, index: number): void {
    this.name = `${layerId}_${index}`;
    this._weight = Math.random();
  }

  public cost(expected: number): number {
    return SharedFunctions.costFunction(expected, this.activatedValue);
  }

  public prediction(x: number): number {
    this.input = this._weight * x;
    return this.output;
  }

  private log = (logLine: string, args: string[] = []): void => {
    if (!this.debug) {
      return;
    }

    console.log(logLine, ...args);
  };
}
