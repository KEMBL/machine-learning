import { Layer } from './';

export class Network {
  public debug = false;

  /** criteria to end learning */
  public error = 0.0001;

  /** maximum learn steps to learn  */
  public maxSteps = 2000;

  /** learning step */
  public ldelta = 0.01;

  private layers: Layer[] = [];

  private get lastLayer(): Layer {
    return this.layers[this.layers.length - 1];
  }

  constructor(maxSteps: number, error: number, ldelta: number) {
    this.maxSteps = maxSteps;
    this.error = error;
    this.ldelta = ldelta;
  }

  /** Adds new layer */
  addLayer = (neuronsCount: number): void => {
    const layerId = this.layers.length + 1;
    const layer = new Layer(layerId, neuronsCount);
    this.layers.push(layer);
  };

  /** Returns output of the last layer */
  result = (): number[] => {
    const lastLayer = this.layers[this.layers.length - 1];
    return lastLayer.result();
  };

  /** Makes learning cycles */
  learn = (inputArray: number[], outputArray: number[]): void => {
    this.layers[0].setOutput(inputArray);
    for (let i = 0; i <= this.maxSteps; i++) {
      const error = this.learnStep(outputArray);
      if (error <= this.error) {
        break;
      }
    }
  };

  /**
   * Performs one learning step
   */
  private learnStep = (outputArray: number[]): number => {
    let error = 1;
    for (let i = 0; i <= this.maxSteps; i++) {
      this.propagate();

      // error find
      error = this.findStepError(outputArray);
      if (error <= this.error) {
        return error;
      }

      // new weights count
      this.backPropagation();
    }

    return error;
  };

  /**
   * Propagate input values through all network
   */
  private propagate = (): void => {
    let previousLayer: Layer;
    for (const layer of this.layers) {
      if (!layer.isFirst) {
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        layer.propagate(previousLayer!);
      }
      previousLayer = layer;
    }
  };

  /**
   * Searches of how big network result error is
   */
  private findStepError = (outputArray: number[]): number => {
    return this.lastLayer.cost(outputArray);    
  };

  /**
   * Count new weights
   */
  private backPropagation = (): void => {
    let previousLayer: Layer;
    for (const layer of this.layers.reverse()) {
      if (!layer.isFirst) {
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        layer.backPropagate(previousLayer!);
      }
      previousLayer = layer;
    }
  };
}
