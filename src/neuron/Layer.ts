import { Neuron } from './';

/**
 * One neurons layer
 */
export class Layer {
  public debug = false;

  // public get isFirst(): boolean {
  //   return this.layerId === 0;
  // }

  public neurons: Neuron[] = [];

  constructor(public layerId: number, private neuronsAmount: number) {
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
    for (let i = 0; i < this.neurons.length; i++) {
      this.propagateNeuron(this.neurons[i], sourceLayer);
      this.neurons[i].prediction();
    }
  };

  /**
   * Takes layer's neuron and feed it with all income signals
   * @param neuron
   */
  private propagateNeuron = (neuron: Neuron, sourceLayer: Layer): void => {
    for (let i = 0; i < sourceLayer.neurons.length; i++) {
      neuron.propagate(i, neuron.output);
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

  /** Receives values of errors on the next layer neurons */
  countErrors = (
    nextLayerOutputArray: number[],
    nextLayer?: Layer
  ): number[] => {
    if (this.layerId === 0) {
      return [];
    }

    const errorWeights: number[] = [];
    for (let i = 0; i < this.neurons.length; i++) {
      if (nextLayer === undefined) {
        this.neurons[i].propagationError = this.neurons[i].cost(
          nextLayerOutputArray[i]
        );
      } else {
        this.neurons[i].propagationError = nextLayer.getWeightError(i);
      }

      errorWeights[i] = this.neurons[i].propagationError;
    }

    return errorWeights;
  };

  /**
   * Collects sum of all errors on the given weight index
   */
  private getWeightError = (inputId: number): number => {
    let error = 0;
    for (let i = 0; i < this.neurons.length; i++) {
      error += this.neurons[i].weightError(inputId);
    }
    return error;
  };

  correctWeights = (learningDelta: number): void => {
    for (let i = 0; i < this.neurons.length; i++) {
      this.neurons[i].correctWeights(learningDelta);
    }
  };
}
