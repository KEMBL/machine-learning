import { Neuron } from './';

/**
 * One neurons layer
 */
export class Layer {
  public debug = false;

  public get isFirst(): boolean {
    return this.layerId === 0;
  }
  private neurons: Neuron[] = [];

  constructor(private layerId: number, private neuronsAmount: number) {
    this.init();
  }

  private init = (): void => {
    this.neurons = [];
    for (let i = 0; i < this.neuronsAmount; i++) {
      const neuron = new Neuron();
      neuron.debug = this.debug;
      neuron.init(this.layerId, i);
      this.neurons.push(neuron);
    }
  };

  /**
   * Init layer, used to set output vars in the first layer
   * @param sourceLayer
   */
  public setOutput = (inputVariables: number[]): void => {
    if (this.layerId !== 0) {
      console.warn(`Init: Current layer ${this.layerId} is nor input layer!`);
    }
    for (let i = 0; i <= this.neurons.length; i++) {
      this.neurons[i].output = inputVariables[i];
    }
  };

  /**
   * Propagate previous layer neurons to all current layer neurons
   * @param sourceLayer
   */
  public propagate = (sourceLayer: Layer): void => {
    if (this.layerId === 0) {
      return;
    }
    for (let i = 0; i < sourceLayer.neurons.length; i++) {
      this.propagateNeuron(sourceLayer.neurons[i]);
    }
  };

  /**
   * Takes source neuron and propagate it to all current layer neurons
   * @param sourceNeuron
   */
  private propagateNeuron = (sourceNeuron: Neuron): void => {
    for (let i = 0; i < this.neurons.length; i++) {
      const neuron = this.neurons[i];
      neuron.prediction(sourceNeuron.output);
    }
  };

  result = (): number[] => {
    const resultsList: number[] = [];
    for (let i = 0; i < this.neurons.length; i++) {
      resultsList.push(this.neurons[i].output);
    }
    return resultsList;
  };

  cost = (outputArray: number[]): number => {
    let cost = 0;
    for (let i = 0; i < this.neurons.length; i++) {
      cost += this.neurons[i].cost(outputArray[i]);
    }
    return cost / (2 * this.neurons.length);
  };

  backPropagate = (nextLayer: Layer): void => {
    //
    //   // new weight
    //   const newWeight = weight + this.learningDelta * cost;
    //   const arrow = cost > 0 ? 'i' : 'v';
    //   const deltaWeight = weight - newWeight;
    //   this.log(
    //     `i:${i}, ${arrow}, cost:${fnz(cost)},  w:${fnz(newWeight)}, dw:${fnz(
    //       deltaWeight
    //     )}, v:${fnz(neuron.output)}, gv:${fnz(prediction)}`
    //   );
    //   if (Math.abs(cost) < this.error) {
    //     break;
    //   }
    //   weight = newWeight;
  };
}
