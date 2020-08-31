import { Layer, StringFunctions } from './';

// shortcut to rounding function
// eslint-disable-next-line no-unused-vars, @typescript-eslint/no-unused-vars
const fnz = StringFunctions.fnz;

/**
 * Feedforward network with back propagation
 */
export class Network {
  public static currentStep = 0;

  private debug = false;
  private name = 'Nt ';

  /** criteria to end learning */
  public maxError = 0.0001;

  /** maximum learn steps to learn  */
  public maxSteps = 2000;

  /** learning step */
  public ldelta = 0.01;

  private layers: Layer[] = [];

  private lastLayer!: Layer;

  constructor(
    inputs: number,
    maxSteps: number,
    maxCostError: number,
    ldelta: number,
    debug?: boolean
  ) {
    this.maxSteps = maxSteps;
    this.maxError = maxCostError;
    this.ldelta = ldelta;
    this.debug = !!debug;

    this.addLayer(inputs); // inuts
  }

  /** Adds new layer */
  addLayer = (neuronsCount: number): void => {
    const layerId = this.layers.length;
    const layer = new Layer(layerId, neuronsCount, this.debug);
    this.layers.push(layer);
    this.lastLayer = layer;
  };

  /** Returns output of the last layer */
  output = (): number[] => {
    return this.lastLayer.output();
  };

  /** Makes learning cycles */
  learn = (inputArray: number[], outputArray: number[]): void => {
    this.layers[0].setOutput(inputArray);
    for (let i = 0; i < this.maxSteps; i++) {
      this.log(`Learn step ${i}`);
      Network.currentStep = i;
      const error = this.learnStep(outputArray);
      if (error <= this.maxError) {
        break;
      }
    }
  };

  /**
   * Performs one learning step
   */
  private learnStep = (outputArray: number[]): number => {
    let error = 1;
    this.propagate();

    // look at error value
    error = this.findStepError(outputArray);
    this.log(`Res1`, fnz(error), '<=?', this.maxError);
    if (error <= this.maxError) {
      this.log(`Res: ${fnz(error)} < ${this.maxError}`);
      return error;
    }

    // new weights count
    this.backPropagation(outputArray);

    console.log('Step weights', this.getWeights());

    return error;
  };

  /**
   * Propagate input values through all network
   */
  private propagate = (): void => {
    let previousLayer: Layer | undefined;
    for (const layer of this.layers) {
      if (previousLayer !== undefined) {
        layer.propagate(previousLayer);
      }
      previousLayer = layer;
    }
  };

  /**
   * Searches of how big network result error is
   */
  public findStepError = (outputArray: number[]): number => {
    const cost = this.lastLayer.cost(outputArray);
    this.log(`Cost error search`, fnz(cost));
    return cost;
  };

  /**
   * Count new weights
   */
  private backPropagation = (outputArray: number[]): void => {
    this.log(`Back propagation`);
    let previousLayer: Layer | undefined = undefined;
    let nextLayerOutputArray = outputArray;
    for (const layer of this.layers.slice().reverse()) {
      nextLayerOutputArray = layer.countErrors(
        nextLayerOutputArray,
        previousLayer
      );
      previousLayer = layer;
    }

    for (const layer of this.layers) {
      layer.correctWeights(this.ldelta);
    }
  };

  public initWeights = (weights: number[][][]): void => {
    this.log('Nw', weights);
    for (let i = 1; i < this.layers.length; i++) {
      this.layers[i].initWeights(weights[i - 1]);
    }
  };

  /** Debug method. Allows to set weights directly */
  public getWeights = (): number[][][] => {
    // this.log('GNe', weights);
    const weights: number[][][] = [];
    for (let i = 1; i < this.layers.length; i++) {
      weights.push(this.layers[i].getWeights());
    }
    return weights;
  };

  private log = (logLine: string, ...args: unknown[]): void => {
    if (!this.debug) {
      return;
    }

    StringFunctions.log(`${this.name}: ${logLine}`, ...args);
  };
}
