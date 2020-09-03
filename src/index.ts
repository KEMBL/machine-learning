import { LayerConfig } from './neuron/models';
import { Configuration, Network } from './neuron';
import { Log } from './services';
import { Verbosity } from './models';

class Program {
  constructor() {
    const started = new Date();
    Log.log('Programm started');

    Log.verbosity = Verbosity.Warning;
    Configuration.bias = 1;
    Configuration.activationType = 'ReLU'; // default
    Configuration.useCostFunction = 'Default';

    // Regression
    const networkInputs = [1, 0];
    const targetOutputs = [1];
    const maximumCostError = 0.0001;
    const maxLearningSteps = 10000;
    const learningDelta = 0.1;
    const layersConfig: LayerConfig[] = [
      { neurons: 2 },
      { neurons: 2 },
      { neurons: 1, activationType: 'Sigmoid' }
    ];

    // Fill in arrays if want to start not from random weights
    // Neurons:  XYZ  X - source output, Y - layer row   Z - input Layer
    // Debug. prefill weights
    //  [ [layer1], [layer2], ..., [[neuron1], [neuron2], ... ], [[[weight1, weight2, ...]], [[weight1, weight2, ...]], ...], [neuron2], ... ]  ]
    const weights: number[][][] = [];
    // const weights: number[] = [
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

    // const weights = [
    //   [ [ 12.073027175758078, -0.42 ], [ 11.29143338568982, 0.38 ] ],
    //   [
    //     [ 2.5379574472175412, 2.060681210357274 ],
    //     [ 4.114487335508431, 4.26457245636459 ]
    //   ],
    //   [ [ 2.11694803045532, 2.897016751994774 ] ]
    // ];

    const network = new Network(
      layersConfig[0].neurons,
      maxLearningSteps,
      maximumCostError,
      learningDelta
    ); // error, ldelta, maxSteps

    for (const layerConfig of layersConfig) {
      network.addLayer(layerConfig); // make neurons / init them / etc
    }

    if (weights.length > 0) {
      network.initWeights(weights);
    }

    network.learn(networkInputs, targetOutputs); // propagate / errorcost / weig\hts correction (back propagation)
    //new Network().testNeuron();
    const result = network.output();
    Log.log('Programm finished', result, targetOutputs);
    Log.log('Result weights', network.getWeights());
    Log.log('Last step', Network.currentStep);
    Log.log('Error cost', network.findStepError(targetOutputs));
    Log.log(
      `Finished in`,
      new Date().getSeconds() - started.getSeconds(),
      'seconds'
    );
  }
}

new Program();
