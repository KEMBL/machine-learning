/**
 * Main settings fo the ML system
 */
class Configuration {
  public bias = 1;
  /** Activation function */
  public activationType = 'Identity';
  /** Cost function */
  public useCostFunction = 'Squared';
}

const configuration = new Configuration();
export { configuration as Configuration };
