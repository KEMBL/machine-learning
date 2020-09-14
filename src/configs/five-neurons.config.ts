import { Network } from '../neuron';
import {
  ActivationType,
  CostFunctionType,
  LayerConfig,
  NetworkConfig
} from '../neuron/models';
import { Log } from '../services';

export class FiveNeurons implements NetworkConfig {
  bias = 1;
  activationFunction: ActivationType = 'ReLU';
  costFunction: CostFunctionType = 'Squared';
  inputsAmount = 2;
  networkInputs = [[1, 0]];
  targetOutputs = [[0.5]];
  learnigSamplesCount = this.networkInputs.length;
  maximumCostError = 0.0001;
  maxEpochsCount = 10000;
  learningDelta = 0.1;
  layersConfig: LayerConfig[] = [
    { neurons: 2 },
    { neurons: 2 },
    { neurons: 1, activationType: 'Sigmoid' }
  ];
  startWeights: number[][][] = [];
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

  constructor() {
    Log.info('Config FiveNeurons', 'config');
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

  isInputGenerated = !!this.generator;
  test = (network: Network): void => {
    Log.log('');
    Log.log('Prediction');
    for (let i = 0; i < this.networkInputs.length; i++) {
      const error = network.predict(
        this.networkInputs[i],
        this.targetOutputs[i]
      );
      Log.log(
        `Step ${i + 1}, Error cost`,
        error,
        network.output(),
        this.targetOutputs[i]
      );
    }
  };
}
