import { Log } from '../services';
import { Layer, StringFunctions, Configuration } from './';
import { LayerConfig } from './models';

// shortcut to rounding function
// eslint-disable-next-line no-unused-vars, @typescript-eslint/no-unused-vars
const fnz = StringFunctions.fnz;
export interface LearnFunction {
  (inputArray: number[][], outputArray: number[][]): number;
  (
    inputOutputGenerator: (
      inputsAmount: number
    ) => Generator<
      { inputArray: number[]; outputArray: number[] },
      void,
      unknown
    >,
    trainigSamples: number
  ): number;
}

/**
 * Feedforward network with back propagation
 */
export class Network {
  /** criteria to end learning */
  public maxError = 0.0001;

  /** maximum learn cycles  */
  public maxEpochsCount = 2000;

  /** learning step */
  public ldelta = 0.01;

  private inputsAmount = 0;

  private currentEpoch = 0;

  private layers: Layer[] = [];

  private lastLayer!: Layer;

  private moduleName = 'Nt';

  constructor(
    inputsAmount: number,
    maxEpochsCount: number,
    maxCostError: number,
    ldelta: number
  ) {
    this.inputsAmount = inputsAmount;
    this.maxEpochsCount = maxEpochsCount;
    this.maxError = maxCostError;
    this.ldelta = ldelta;

    this.addLayer({ neurons: this.inputsAmount }); // inuts
  }

  /** Adds new layer */
  public addLayer = (config: LayerConfig): void => {
    const selectedFunction = config.activationType
      ? config.activationType
      : Configuration.activationType;
    const layerId = this.layers.length;
    const layer = new Layer(layerId, config.neurons, selectedFunction);
    this.layers.push(layer);
    this.lastLayer = layer;
  };

  /** Returns output of the last layer */
  public output = (): number[] => {
    return this.lastLayer.output();
  };

  /** Performn epochs untill max epochs count or error is low */
  public train: LearnFunction = (
    inputArrayOrGenerator: any,
    outputArrayOrSamplesAmount?: any
  ): number => {
    let error = 0;
    for (let i = 0; i < this.maxEpochsCount; i++) {
      this.currentEpoch = i + 1;
      error = this.performEpoch(
        inputArrayOrGenerator,
        outputArrayOrSamplesAmount
      );
      // console.log(
      //   'Epoch Math.abs(error) <= this.maxError',
      //   error,
      //   this.maxError,
      //   i
      // );
      if (Math.abs(error) <= this.maxError) {
        Log.debug(`Found solution at epoch: ${i}`, this.moduleName);
        break;
      }
    }

    Log.debug(
      `Epoch: ${this.maxEpochsCount}, error ${fnz(error)} < ${this.maxError}`,
      this.moduleName
    );

    Log.prefix = undefined;
    return error;
  };

  /** Start epoch with correspondent samples */
  public performEpoch: LearnFunction = (
    inputArrayOrGenerator: any,
    outputArrayOrSamplesAmount?: any
  ): number => {
    let epochError = 0;
    if (Array.isArray(inputArrayOrGenerator)) {
      if (
        !inputArrayOrGenerator ||
        inputArrayOrGenerator.length === 0 ||
        inputArrayOrGenerator.length !== outputArrayOrSamplesAmount.length
      ) {
        Log.throw(
          'Input array empty or amount of input examples does not match amount of output exaples',
          this.moduleName,
          inputArrayOrGenerator?.length,
          outputArrayOrSamplesAmount?.length
        );
      }

      epochError = this.performTrainingFromStaticSample(
        inputArrayOrGenerator,
        outputArrayOrSamplesAmount
      );
    } else {
      epochError = this.performTrainingFromSampleGenerator(
        inputArrayOrGenerator,
        outputArrayOrSamplesAmount
      );
    }

    return epochError;
  };

  /** Makes learning cycles with input/output data taken from const arrays */
  public performTrainingFromStaticSample = (
    inputArray: number[][],
    outputArray: number[][]
  ): number => {
    const samplesAmount = inputArray.length;
    let samplesError = 0;
    for (let i = 0; i < samplesAmount; i++) {
      Log.prefix = `${i}.${this.currentEpoch}`;
      Log.debug(
        `Static sample ${i + 1}`,
        this.moduleName,
        inputArray[i],
        outputArray[i]
      );
      samplesError += this.performPredictionAndBackPropagation(
        inputArray[i], // TODO: 4) How is better to represent examples from the learning set? Random or one by one?
        outputArray[i]
      );
    }
    return samplesError;
  };

  /** Makes learning cycles with trining input/output data taken from generator */
  public performTrainingFromSampleGenerator = (
    // inputOutputGenerator: {next:() => { inputArray: number[]; outputArray: number[] }}
    inputOutputGenerator: (
      inputsAmount: number
    ) => Generator<
      { inputArray: number[]; outputArray: number[] },
      void,
      unknown
    >,
    samplesNum: number
  ): number => {
    let samplesError = 0;
    const trainigDataGenerator = inputOutputGenerator(this.inputsAmount);
    for (let i = 0; i < samplesNum; i++) {
      const trainigSet = trainigDataGenerator.next();
      if (trainigSet.value) {
        Log.prefix = `${i}.${this.currentEpoch}`;
        Log.debug(
          `Generated sample ${i + 1}`,
          this.moduleName,
          trainigSet.value
        );
        const sampleError = this.performPredictionAndBackPropagation(
          trainigSet.value.inputArray,
          trainigSet.value.outputArray
        );
        // console.log('se', i, sampleError, samplesError, trainigSet.value);
        samplesError += sampleError;
      } else {
        Log.throw(
          `Sample num: ${i}. Cannot extract value from set generator`,
          this.moduleName
        );
      }
    }

    return samplesError;
  };

  /** Configs and makes one learcning step */
  // public configAndTrain = (
  //   sampleNum: number,
  //   inputArray: number[],
  //   outputArray: number[]
  // ): number => {
  //   Log.debug(`Sample learn sample ${sampleNum + 1}`, this.moduleName);
  //   return this.performPredictionAndBackPropagation(inputArray, outputArray);
  // };

  /**
   * Perform prediction over given sample and update the model parameters
   */
  private performPredictionAndBackPropagation = (
    inputSampleArray: number[],
    awaitedResultArray: number[]
  ): number => {
    this.propagate(inputSampleArray);

    // new weights count
    this.backPropagation(awaitedResultArray);

    // look at error value
    const error = this.findPropagationError(awaitedResultArray);
    Log.debug(
      `Sample prediction error:`,
      this.moduleName,
      fnz(error),
      '<=?',
      this.maxError
    );

    return error;
  };

  /**
   * Propagate input values through all network
   */
  private propagate = (inputSampleArray: number[]): void => {
    this.layers[0].setOutput(inputSampleArray); // fill in network inputs
    let previousLayer: Layer | undefined;
    for (const layer of this.layers) {
      if (previousLayer !== undefined) {
        layer.propagate(previousLayer);
      }
      previousLayer = layer;
    }
  };

  /**
   * Make one time prediction with returning of prediction error
   */
  public predict = (
    inputSampleArray: number[],
    awaitedResultArray: number[]
  ): number => {
    this.propagate(inputSampleArray);
    return this.findPropagationError(awaitedResultArray);
  };

  /**
   * Searches of how big network result error is
   */
  public findPropagationError = (outputArray: number[]): number => {
    const cost = this.lastLayer.cost(outputArray);
    Log.debug(`Cost error search`, this.moduleName, fnz(cost));
    return cost;
  };

  /**
   * Returns current network error
   */
  public networkError = (): number => {
    return this.lastLayer.costError();
  };

  /**
   * Returns current network error
   */
  public epochsCount = (): number => {
    return this.currentEpoch;
  };

  /**
   * Count new weights
   */
  private backPropagation = (outputArray: number[]): void => {
    Log.debug(`Back propagation`, this.moduleName);
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

    Log.debug('BackPropagation weights', this.moduleName, this.getWeights());
  };

  public initWeights = (weights: number[][][]): void => {
    Log.debug('Nw', this.moduleName, weights);
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
}
