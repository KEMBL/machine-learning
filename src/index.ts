import { Configuration, Network } from './neuron';
import { Log } from './services';
import { Verbosity } from './models';
import { SinusGenerated } from './configs';

class Program {
  constructor() {
    const startedTime = new Date();
    Log.log('Programm started');
    Log.verbosity = Verbosity.Warning;

    //const conf = new FiveNeurons();
    const conf = new SinusGenerated();
    Configuration.bias = conf.bias;
    Configuration.activationType = conf.activationFunction;
    Configuration.useCostFunction = conf.costFunction;

    // Regression
    const inputsAmount = conf.inputsAmount;

    // if(startedTime.getTime() !==1) return;

    // 2,2,1- 41 sec
    // 3,3,1- 60 sec
    // 3,3,3,1 - 80 sec

    const networkInputs = conf.networkInputs;
    const targetOutputs = conf.targetOutputs;
    const maximumCostError = conf.maximumCostError;
    const maxEpochsCount = conf.maxEpochsCount;
    const learningDelta = conf.learningDelta;
    const layersConfig = conf.layersConfig;

    // Fill in arrays if want to start not from random weights
    // Neurons:  XYZ  X - source output, Y - layer row   Z - input Layer
    // Debug. prefill weights
    //  [ [layer1], [layer2], ..., [[neuron1], [neuron2], ... ], [[[weight1, weight2, ...]], [[weight1, weight2, ...]], ...], [neuron2], ... ]  ]
    const weights: number[][][] = conf.startWeights;

    // const weights = [
    //   [ [ 12.073027175758078, -0.42 ], [ 11.29143338568982, 0.38 ] ],
    //   [
    //     [ 2.5379574472175412, 2.060681210357274 ],
    //     [ 4.114487335508431, 4.26457245636459 ]
    //   ],
    //   [ [ 2.11694803045532, 2.897016751994774 ] ]
    // ];

    const network = new Network(
      inputsAmount,
      maxEpochsCount,
      maximumCostError,
      learningDelta
    ); // error, ldelta, maxSteps

    for (const layerConfig of layersConfig) {
      network.addLayer(layerConfig); // make neurons / init them / etc
    }

    if (weights.length > 0) {
      network.initWeights(weights);
    }

    if (conf.isInputGenerated) {
      network.train(conf.generator.bind(this), conf.learnigSamplesCount); // propagate / errorcost / weig\hts correction (back propagation)
    } else {
      // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
      network.train(networkInputs!, targetOutputs!); // propagate / errorcost / weig\hts correction (back propagation)
    }

    const result = network.output();
    Log.log('Programm finished', result, targetOutputs);
    Log.log('Result weights', network.getWeights());
    Log.log('Error cost', network.networkError());
    Log.log('Epochs', network.epochsCount());
    Log.log(
      `Finished in`,
      (new Date().getTime() - startedTime.getTime()) * 0.001,
      'seconds'
    );
    if (conf.test) {
      conf.test(network);
    }

    //if (startedTime.getTime() === -1) {
    // Log.log('');
    // Log.log('Prediction');
    // for (let i = 0; i < networkInputs.length; i++) {
    //   const error = network.predict(networkInputs[i], targetOutputs[i]);
    //   Log.log(
    //     `Step ${i + 1}, Error cost`,
    //     error,
    //     network.output(),
    //     targetOutputs[i]
    //   );
    // }
    //   const generator = generatorSinus(inputsAmount);
    //   for (let i = 0; i < 3; i++) {
    //     const sample = generator.next();
    //     if (sample.value) {
    //       const error = network.predict(
    //         sample.value.inputArray,
    //         sample.value.outputArray
    //       );
    //       Log.log(
    //         `Step ${i + 1}, Error cost`,
    //         error,
    //         network.output(),
    //         sample.value.outputArray
    //       );
    //     }
    //   }
    //}
  }
}

new Program();
