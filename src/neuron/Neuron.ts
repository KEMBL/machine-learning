import { Configuration } from './configuration/Configuration';

export class Neuron {
  private name = '';
  private inputValue = 0;
  private activatedValue = 0;

  public get output(): number {
    return this.activatedValue;
  }

  public set input(x: number) {
    this.inputValue = x + Configuration.bias;
    this.activatedValue = Configuration.activationFunction(this.inputValue);
    console.log(`nr ${this.name} v: ${this.inputValue} -> ${this.activatedValue}, in: ${x}`);
  }

  public init(layerId: number, index: number): void {
    this.name = `${layerId}_${index}`;
  }

  public cost(expected: number): number {
    return Configuration.costFunction(expected, this.activatedValue);
  }

  public prediction(weight: number, x: number): number {
    this.input = weight * x;
    return this.output;
  }
}
