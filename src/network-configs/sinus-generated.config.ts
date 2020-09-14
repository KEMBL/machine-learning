import { Network } from '../neuron';
import {
  ActivationType,
  CostFunctionType,
  LayerConfig,
  NetworkConfig
} from '../neuron/models';
import { Log } from '../services';

/**
 * Regression. Feedforward neural network with backpropagation config
 * Inputs and outputs should be taken by network from from generator function
 */
export class SinusGenerated implements NetworkConfig {
  name = 'Sinus prediction';
  bias = 1;
  activationFunction: ActivationType = 'ReLU';
  costFunction: CostFunctionType = 'Squared';
  inputsAmount = 1;
  networkInputs?: number[][];
  targetOutputs?: number[][];
  learnigSamplesCount = 360;
  maximumCostError = 0.0001;
  maxEpochsCount = 10000;
  learningDelta = 0.1;
  layersConfig: LayerConfig[] = [
    { neurons: 3 },
    { neurons: 3 },
    { neurons: 3 },
    { neurons: 1, activationType: 'Sigmoid' }
  ];
  startWeights?: number[][][];

  constructor() {
    Log.info(this.name, 'config');
    // const gen = generator(2);
    // console.log(gen.next());
    // console.log(gen.next());
    // console.log(gen.next());
    // console.log(gen.next());
    // console.log(gen.next());
    // console.log(gen.next());
    // console.log(gen.next());
    // console.log(gen.next());
    // console.log(gen.next());}
  }

  *generator(
    inputsAmount: number
  ): Generator<
    {
      inputArray: number[];
      outputArray: number[];
    },
    void,
    unknown
  > {
    let angle = 1; // degress
    const max = 360;
    while (true) {
      const set = {
        inputArray: Array<number>(inputsAmount).fill(0),
        outputArray: Array<number>(1)
      };
      const degress = (angle * Math.PI) / 180;
      set.inputArray[0] = degress;
      set.outputArray = [Math.sin(degress)];
      yield set;
      angle = ++angle % max;
    }
  }

  //isInputGenerated = !!this.generator;

  test = (network: Network): void => {
    Log.log('');
    Log.log('Prediction test');

    const generator = this.generator(this.inputsAmount);
    for (let sampleId = 0; sampleId < 3; sampleId++) {
      const sample = generator.next();
      if (sample.value) {
        const error = network.predict(
          sample.value.inputArray,
          sample.value.outputArray
        );
        Log.log(
          `Sample ${sampleId + 1}, Error cost`,
          error,
          network.output(),
          sample.value.outputArray
        );
      }
    }
  };
}
