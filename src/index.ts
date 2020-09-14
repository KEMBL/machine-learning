import nopt from 'nopt';

import { Configuration, Network, NetworkConfig } from './neuron';
import { Log } from './services';
import { Verbosity } from './models';
import { SinusGenerated, FiveNeurons, DivisionByTwo } from './network-configs';

/**
 * Application entry point
 */
class Program {
  constructor() {
    const opts = this.readOpts();
    const startedTime = new Date();
    Log.log('Program is started');

    Log.verbosity = Verbosity.Debug;

    let conf: NetworkConfig;
    switch (opts['func']) {
      case 'sin':
        conf = new SinusGenerated();
        break;
      case 'div':
        conf = new DivisionByTwo();
        break;
      default:
        // five
        conf = new FiveNeurons();
        break;
    }

    Configuration.bias = conf.bias;
    Configuration.activationType = conf.activationFunction;
    Configuration.useCostFunction = conf.costFunction;

    const inputsAmount = conf.inputsAmount;
    const networkInputs = conf.networkInputs;
    const targetOutputs = conf.targetOutputs;
    const maximumCostError = conf.maximumCostError;
    const maxEpochsCount = conf.maxEpochsCount;
    const learningDelta = conf.learningDelta;
    const layersConfig = conf.layersConfig;

    const network = new Network(
      inputsAmount,
      maxEpochsCount,
      maximumCostError,
      learningDelta
    );

    network.addLayers(layersConfig); // make neurons / init them / etc

    if (conf.startWeights) {
      // mostly is used for debug
      network.initWeights(conf.startWeights);
    }

    if (conf.generator) {
      network.train(conf.generator, conf.learnigSamplesCount);
    } else {
      // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
      network.train(networkInputs!, targetOutputs!);
    }

    const result = network.output();
    const epochsCount = network.epochsCount();
    Log.log('Program is finished', result);
    Log.log(
      'Epochs',
      epochsCount,
      `${epochsCount < conf.maxEpochsCount ? 'Success' : 'Fail'}`
    );
    Log.log('Error cost', network.networkErrorSignal());
    Log.log('Result weights', network.getWeights());

    Log.log(
      `Finished in`,
      (new Date().getTime() - startedTime.getTime()) * 0.001,
      'seconds'
    );

    if (conf.test) {
      Log.log(`Startting prediction test`);
      conf.test(network);
    }
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private readOpts = (): any => {
    const knownOpts = { func: [String, null] };
    return nopt(knownOpts, {}, process.argv, 2);
  };
}

new Program();
