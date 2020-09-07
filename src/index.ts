import { LayerConfig } from './neuron/models';
import { Configuration, Network } from './neuron';
import { Log } from './services';
import { Verbosity } from './models';

class Program {
  constructor() {
    const startedTime = new Date();
    Log.log('Programm started');

    Log.verbosity = Verbosity.Warning;
    Configuration.bias = 1;
    Configuration.activationType = 'ReLU'; // default
    Configuration.useCostFunction = 'Squared';

    // Regression
    const inputsAmount = 1;
    function* generatorSinus(
      inputsAmount: number
    ): Generator<
      {
        inputArray: number[];
        outputArray: number[];
      },
      void,
      unknown
    > {
      let angle = 1; // degress
      const max = 360;
      while (true) {
        const set = {
          inputArray: Array<number>(inputsAmount).fill(0),
          outputArray: Array<number>(1)
        };
        const degress = (angle * Math.PI) / 180;
        set.inputArray[0] = degress;
        set.outputArray = [Math.sin(degress)];
        yield set;
        angle = ++angle % max;
      }
    }

    // const gen = generatorSinus(2);
    // console.log(gen.next());
    // console.log(gen.next());
    // console.log(gen.next());
    // console.log(gen.next());
    // console.log(gen.next());
    // console.log(gen.next());
    // console.log(gen.next());
    // console.log(gen.next());
    // console.log(gen.next());

    // if(startedTime.getTime() !==1) return;

    // 2,2,1- 41 sec
    // 3,3,1- 60 sec
    // 3,3,3,1 - 80 sec

    //const networkInputs = [[1, 0]];
    const targetOutputs = [[0.5]];
    const maximumCostError = 0.0001;
    const maxEpochsCount = 10000;
    const learningDelta = 0.1;
    const layersConfig: LayerConfig[] = [
      { neurons: 3 },
      { neurons: 3 },
      { neurons: 3 },
      // { neurons: 20 },
      //       { neurons: 3 },
      { neurons: 1, activationType: 'Sigmoid' }
      //{ neurons: 1 }
    ];

    // Fill in arrays if want to start not from random weights
    // Neurons:  XYZ  X - source output, Y - layer row   Z - input Layer
    // Debug. prefill weights
    //  [ [layer1], [layer2], ..., [[neuron1], [neuron2], ... ], [[[weight1, weight2, ...]], [[weight1, weight2, ...]], ...], [neuron2], ... ]  ]
    const weights: number[][][] = [];
    // const weights: number[][][] = [
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

    //network.train(networkInputs, targetOutputs); // propagate / errorcost / weig\hts correction (back propagation)
    network.train(generatorSinus, 360); // propagate / errorcost / weig\hts correction (back propagation)
    //new Network().testNeuron();
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

    if (startedTime.getTime() === -1) {
      Log.log('');
      Log.log('Prediction');
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
    }
  }
}

new Program();
