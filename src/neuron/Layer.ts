import { Log } from '../services';
import { ActivationType } from './models';
import { Neuron } from './';

// shortcut to rounding function
// eslint-disable-next-line no-unused-vars, @typescript-eslint/no-unused-vars
// const _fnz = StringFunctions.fnz;

/**
 * One neurons layer
 */
export class Layer {
  public neurons: Neuron[] = [];

  private moduleName = '';
  private activationType: ActivationType = 'ReLU';
  /** Layer absolute error signals sum */
  private layerErrorSignal = 0;

  constructor(
    public layerId: number,
    private neuronsAmount: number,
    activationType?: ActivationType
  ) {
    if (activationType) {
      this.activationType = activationType;
    }
    this.init();
  }

  public toString = (): string => {
    return this.moduleName;
  };

  private init = (): void => {
    this.neurons = [];
    this.moduleName = `Lr ${this.layerId}`;
    Log.debug(
      `Config: neuronsAmount ${this.neuronsAmount}, activation: ${this.activationType}`,
      this.moduleName
    );
    for (let neuronId = 0; neuronId < this.neuronsAmount; neuronId++) {
      const neuron = new Neuron(this.layerId, neuronId, this.activationType);
      this.neurons.push(neuron);
    }
  };

  /** Allows to modify weighs of neurons for debug purposes */
  public initWeights = (weights: number[][]): void => {
    // this.log('Lw', weights);
    for (let neuronId = 0; neuronId < this.neurons.length; neuronId++) {
      const neuron = this.neurons[neuronId];
      neuron.initWeights(weights[neuronId]);
    }
  };

  /** Debug method. Allows to set weights directly */
  public getWeights = (): number[][] => {
    // this.log('GNe', weights);
    const weights: number[][] = [];
    for (let neuronId = 0; neuronId < this.neurons.length; neuronId++) {
      const neuron = this.neurons[neuronId];
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
      Log.debug(
        `WARN: Current layer ${this.layerId} is not an input layer!`,
        this.moduleName
      );
    }
    for (let neuronId = 0; neuronId < this.neurons.length; neuronId++) {
      this.neurons[neuronId].output = inputVariables[neuronId];
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
    for (let neuronId = 0; neuronId < this.neurons.length; neuronId++) {
      Log.debug(
        `propagate`,
        this.moduleName,
        neuronId,
        this.neurons.length,
        `${sourceLayer}`
      );
      this.propagateNeuron(this.neurons[neuronId], sourceLayer);
      this.neurons[neuronId].prediction();
    }
  };

  /**
   * Takes layer's neuron and feed it with all income signals
   * @param neuron
   */
  private propagateNeuron = (neuron: Neuron, sourceLayer: Layer): void => {
    Log.debug(
      `propagateNeuron`,
      this.moduleName,
      sourceLayer.neurons.length,
      `${sourceLayer}`
    );
    for (let neuronId = 0; neuronId < sourceLayer.neurons.length; neuronId++) {
      neuron.propagate(neuronId, sourceLayer.neurons[neuronId].output);
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

  /** Sum of error signals of ll layer neurons */
  // public findLayerErrorSignalSum = (outputArray: number[]): number => {
  //   let cost = 0;
  //   for (let neuronId = 0; neuronId < this.neurons.length; neuronId++) {
  //     cost += this.neurons[neuronId].findErrorSignal(outputArray[neuronId]);
  //   }
  //   this.layerErrorCost = cost;// / this.neurons.length;
  //   // this.log(`Lec: ${fnz(layerErrorCost)}`);
  //   return this.layerErrorCost;
  // };

  public errorSignalSum = (): number => {
    return this.layerErrorSignal;
  };

  /** Receives values of error signals of the next layer neurons and counts self error signals */
  public countErrorSignals = (
    nextLayerErrorSignalsArray: number[],
    nextLayer?: Layer
  ): number[] => {
    Log.debug(`CountErrors`, this.moduleName);
    if (this.layerId === 0) {
      // the first layer - nothing to do
      return [];
    }

    this.layerErrorSignal = 0;
    const errorSignals: number[] = [];
    for (let neuronId = 0; neuronId < this.neurons.length; neuronId++) {
      let errorSignal = 0;
      if (nextLayer === undefined) {
        // last layer counts error signal agains expected output values
        errorSignal = this.neurons[neuronId].findErrorSignal(
          nextLayerErrorSignalsArray[neuronId]
        );
      } else {
        errorSignal = nextLayer.getWeightError(neuronId);
      }

      if (!isFinite(errorSignal)) {
        Log.throw(
          `neuron ${neuronId +
            1} errorWeight ${errorSignal}, is the last layer ${nextLayer ===
            undefined}`,
          this.moduleName
        );
      }

      this.neurons[neuronId].errorSignal = errorSignal;
      errorSignals[neuronId] = errorSignal;
      this.layerErrorSignal += Math.abs(errorSignal);
    }
    Log.debug(`PropagationError`, this.moduleName, errorSignals);
    return errorSignals;
  };

  /**
   * Collects sum of all errors on the given weight index
   */
  private getWeightError = (inputId: number): number => {
    let error = 0;
    for (let neuronId = 0; neuronId < this.neurons.length; neuronId++) {
      error += this.neurons[neuronId].weightedErrorSignal(inputId);
    }
    return error;
  };

  public correctWeights = (learningDelta: number): void => {
    for (let i = 0; i < this.neurons.length; i++) {
      this.neurons[i].correctWeights(learningDelta);
    }
  };
}
