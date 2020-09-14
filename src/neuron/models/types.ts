import { Network } from '..';

export type ActivationType = 'ReLU' | 'LeakyReLU' | 'Sigmoid' | 'Identity';

export type CostFunctionType = 'Squared' | 'Default';
export interface LayerConfig {
  neurons: number;
  activationType?: ActivationType;
}

/**
 * NeuralnNetwork configuration interface
 */
export interface NetworkConfig {
  bias: number;
  /** default layer cost function */
  activationFunction: ActivationType;
  costFunction: CostFunctionType;
  inputsAmount: number;
  isInputGenerated: boolean;
  generator?: (
    inputsAmount: number
  ) => Generator<
    {
      inputArray: number[];
      outputArray: number[];
    },
    void,
    unknown
  >;
  learnigSamplesCount: number;
  networkInputs?: number[][];
  targetOutputs?: number[][];
  maximumCostError: number;
  maxEpochsCount: number;
  learningDelta: number;
  layersConfig: LayerConfig[];
  startWeights?: number[][][];
  test: (network: Network) => void;
}
