import { Neuron, StringFunctions } from './';

// shortcut to rounding function
// eslint-disable-next-line no-unused-vars, @typescript-eslint/no-unused-vars
// const _fnz = StringFunctions.fnz;

/**
 * One neurons layer
 */
export class Layer {
  private debug = false;
  private name = '';
  public neurons: Neuron[] = [];

  constructor(
    public layerId: number,
    private neuronsAmount: number,
    debug?: boolean
  ) {
    this.debug = !!debug;
    this.init();
  }

  private init = (): void => {
    this.neurons = [];
    this.name = `Lr ${this.layerId}`;
    for (let i = 0; i < this.neuronsAmount; i++) {
      const neuronId = i + 1;
      const neuron = new Neuron(this.layerId, neuronId, this.debug);
      this.neurons.push(neuron);
    }
  };

  /** Allows to modify weighs of neurons for debug purposes */
  public initWeights = (weights: number[][]): void => {
    // this.log('Lw', weights);
    for (let i = 0; i < this.neurons.length; i++) {
      const neuron = this.neurons[i];
      neuron.initWeights(weights[i]);
    }
  };

  /** Debug method. Allows to set weights directly */
  public getWeights = (): number[][] => {
    // this.log('GNe', weights);
    const weights: number[][] = [];
    for (let i = 0; i < this.neurons.length; i++) {
      const neuron = this.neurons[i];
      weights.push(neuron.getWeights());
    }
    return weights;
  };

  /**
   * Init layer, used to set output vars in the first layer
   * @param sourceLayer
   */
  public setOutput = (inputVariables: number[]): void => {
    if (this.layerId !== 0) {
      this.log(`WARN: Current layer ${this.layerId} is not an input layer!`);
    }
    for (let i = 0; i < this.neurons.length; i++) {
      this.neurons[i].output = inputVariables[i];
    }
  };

  /**
   * Propagate previous layer neurons to all current layer neurons
   * @param sourceLayer
   */
  public propagate = (sourceLayer: Layer): void => {
    // this.log(
    //   `Propagate layer ${this.layerId} from layer ${sourceLayer.layerId}`,
    //   this.neurons.length
    // );
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
    // this.log(`propagateNeuron`, sourceLayer.neurons.length);
    for (let i = 0; i < sourceLayer.neurons.length; i++) {
      neuron.propagate(i, sourceLayer.neurons[i].output);
      // neuron.propagate(0, sourceLayer.neurons[i].output);
    }
  };

  public output = (): number[] => {
    const resultsList: number[] = [];
    for (let i = 0; i < this.neurons.length; i++) {
      resultsList.push(this.neurons[i].output);
    }
    return resultsList;
  };

  public cost = (outputArray: number[]): number => {
    let cost = 0;
    for (let i = 0; i < this.neurons.length; i++) {
      cost += this.neurons[i].cost(outputArray[i]);
    }
    const layerErrorCost = cost / (2 * this.neurons.length); // TODO: ? what is the purpose of division  by 2*... ?
    // this.log(`Lec: ${fnz(layerErrorCost)}`);
    return layerErrorCost;
  };

  /** Receives values of errors on the next layer neurons */
  public countErrors = (
    nextLayerOutputArray: number[],
    nextLayer?: Layer
  ): number[] => {
    this.log(`CountErrors`);
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
    this.log(`PropagationError`, errorWeights);
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

  public correctWeights = (learningDelta: number): void => {
    for (let i = 0; i < this.neurons.length; i++) {
      this.neurons[i].correctWeights(learningDelta);
    }
  };

  private log = (logLine: string, ...args: unknown[]): void => {
    if (!this.debug) {
      return;
    }

    StringFunctions.log(`${this.name}: ${logLine}`, ...args);
  };
}
