import { ActivationType, CostType } from '../models';

/**
 * Main settings fo the ML system
 */
export class Configuration {
  /** Allows neurons with 0 inputs still work in the network */
  public static bias = 1;
  /** Activation function */
  public static activationType: ActivationType = 'ReLU';
  /** Cost function */
  public static useCostFunction: CostType = 'Squared';
}
