import { ActivationType, CostType } from '../models';

/**
 * Main settings fo the ML system
 */
export class Configuration {
  public static bias = 1;
  /** Activation function */
  public static activationType: ActivationType = 'ReLU';
  /** Cost function */
  public static useCostFunction: CostType = 'Squared';
}
