import { Network } from '../neuron';
import {
  ActivationType,
  CostFunctionType,
  LayerConfig,
  NetworkConfig
} from '../neuron/models';
import { Log } from '../services';

/**
 * Devide 1 / 2
 * Just a learnig test with one sample
 * Regression. Feedforward neural network with backpropagation config
 * Inputs are predefined in arrays
 */
export class FiveNeurons implements NetworkConfig {
  name = 'Config FiveNeurons';
  bias = 1;
  activationFunction: ActivationType = 'ReLU';
  costFunction: CostFunctionType = 'Default';
  networkInputs = [[1, 0]];
  targetOutputs = [[0.5]];
  inputsAmount = this.networkInputs[0].length;
  learnigSamplesCount = this.networkInputs.length;
  maximumCostError = 0.0001;
  maxEpochsCount = 10000;
  learningDelta = 0.1;
  layersConfig: LayerConfig[] = [
    { neurons: 2 },
    { neurons: 2 },
    { neurons: 1, activationType: 'Sigmoid' }
  ];
  // Fill in arrays if want to start not from random weights
  // Neurons:  XYZ  X - source output, Y - layer row   Z - input Layer
  // Debug. prefill weights
  //  [ [layer1], [layer2], ..., [[neuron1], [neuron2], ... ], [[[weight1, weight2, ...]], [[weight1, weight2, ...]], ...], [neuron2], ... ]  ]
  startWeights?: number[][][];
  //   startWeights: number[][][] = [
  //   [
  //     [0.13, -0.42], // w111, w211
  //     [-0.34, 0.38] // w121, w221
  //   ],
  //   [
  //     [0.25, -0.2], // w112, w212
  //     [0.07, 0.32] // w122, 2222
  //   ],
  //   [[-0.41, 0.12]] // w113, w213
  // ];

  // learned
  // const weights = [
  //   [ [ 12.073027175758078, -0.42 ], [ 11.29143338568982, 0.38 ] ],
  //   [
  //     [ 2.5379574472175412, 2.060681210357274 ],
  //     [ 4.114487335508431, 4.26457245636459 ]
  //   ],
  //   [ [ 2.11694803045532, 2.897016751994774 ] ]
  // ];

  constructor() {
    Log.info(this.name, 'config');
  }

  generator?: (
    inputsAmount: number
  ) => Generator<
    {
      inputArray: number[];
      outputArray: number[];
    },
    void,
    unknown
  > = undefined;

  //isInputGenerated = !!this.generator;
  test = (network: Network): void => {
    Log.log('');
    Log.log('Prediction test');
    const expected = 0.5;
    const error = network.predict([1, 0], [expected]);
    Log.log(
      `Sample 1, input:`,
      1,
      'output:',
      network.output()[0],
      'expected:',
      expected,
      'network error: ',
      error,
      'actual error',
      (expected - network.output()[0]) * 0.5,
      'error diff',
      error - (expected - network.output()[0]) * 0.5
    );
  };
}