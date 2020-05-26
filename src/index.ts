import { Network } from './neuron/Network';

class Program {
  constructor() {
    console.log('Programm started');

    new Network().testNeuron();

    console.log('Programm finished');
  }
}

new Program();
