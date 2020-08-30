import { Configuration, Network } from './neuron';

class Program {
  constructor() {
    console.log('Programm started');

    Configuration.bias = 0;
    Configuration.activationType = 'None';
    Configuration.useCostFunction = 'None';
    const inputs = [1];
    const targetOutputs = [40];

    const error = 0.0001;
    const maxSteps = 2000;
    const ldelta = 0.01;

    const neuronsCount = inputs.length;
    const network = new Network(maxSteps, error, ldelta); // error, ldelta, maxSteps
    network.debug = false;
    network.addLayer(neuronsCount); // make neurons / init them / etc
    network.learn(inputs, targetOutputs); // propagate / errorcost / weig\hts correction (back propagation)
    //new Network().testNeuron();
    const result = network.result();
    console.log('Programm finished', result);
  }
}

new Program();

// class Program1 {
//   constructor() {
//     console.log('Programm started');

//     const inputs: number[] = [1, 50, 10, 0.3, 1, 2, 10, 45];
//     const outputs: number[] = [50, 1, 10, 20, 100, 33, 1, 15];
//     const results: LearningResult[] = [];

//     for (let i = 0; i < inputs.length; i++) {
//       const network = new Network(inputs[i], outputs[i]);
//       network.debug = false;
//       results.push(network.testNeuron(3));
//     }

//     for (let i = 0; i < inputs.length; i++) {
//       console.log(
//         `${i}. steps ${results[i].steps}, cost ${results[i].cost} input ${inputs[i]} * w ${results[i].weight} = ${outputs[i]} `,
//         inputs[i] * results[i].weight,
//         inputs[i] * results[i].weight === outputs[i],
//         Math.round(inputs[i] * results[i].weight) === outputs[i]
//       );
//     }

//     console.log('Programm finished');
//   }
// }
