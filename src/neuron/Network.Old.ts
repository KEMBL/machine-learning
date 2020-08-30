// import { Neuron } from './Neuron';
// import { SharedFunctions } from './configuration/SharedFunctions';
// import { StringFunctions } from './utilities/StringFunctions';
// import { Configuration } from './configuration/Configuration';

// const fnz = StringFunctions.fnz;

// export interface LearningResult {
//   weight: number;
//   steps: number;
//   cost: number;
// }

// export class NetworkOld {
//   public debug = false;

//   /** input value */
//   private input = 1.0;
//   /** value which we want to achieve for given input */
//   private target = 40.0;
//   private error = 0.0001;
//   /** Decreases speed of network adaptation to a given task, makes that process more predictable */
//   private learningDelta = 0.01;
//   private maxSteps = 1000;
//   // private neurons: Neuron[] = [];8

//   //private i: number[];

//   constructor(input?: number, target?: number) {
//     //this.i = [0,1,2];
//     if (!!input || input === 0) {
//       this.input = input;
//     }

//     if (!!target || target === 0) {
//       this.target = target;
//     }
//   }
//   /*
//   private init = (inNum:number, oNum) => {
//     this.neurons = [];
//     for(let i = 0; i < inNum; i++)
//       {
//         this.neurons.push(new Neuron);
//         this.neurons[i].init();
//       }
//   }
// */

//   public testNeuron = (type: number): LearningResult => {
//     switch (type) {
//       case 1:
//         return this.testNeuronGw();
//         break;
//       case 2:
//         return this.testNeuronReal();
//         break;
//       case 3:
//         this.maxSteps = 2000;
//         Configuration.bias = 0;
//         Configuration.activationType = 'None';
//         Configuration.useCostFunction = 'None';
//         return this.testNeuronSimple();
//         break;
//       default:
//         return this.testNeuronUnbounded();
//         break;
//     }
//   };

//   /** simple way to count a new weight as weight + cost * learning_delta */
//   public testNeuronSimple = (): LearningResult => {
//     const neuron = new Neuron();
//     neuron.debug = this.debug;

//     let weight = 1; //Math.random()*5.0;
//     this.log(
//       `init o:${this.target}, w:${weight}, ld:${this.learningDelta}, e:${this.error}`
//     );

//     let i = 0;
//     let cost = 0;
//     for (i = 0; i < this.maxSteps; i++) {
//       // forward propagation
//       const prediction = neuron.prediction(weight, this.input);

//       // error find
//       cost = neuron.cost(this.target);

//       // new weight
//       const newWeight = weight + this.learningDelta * cost;

//       const arrow = cost > 0 ? 'i' : 'v';
//       const deltaWeight = weight - newWeight;

//       this.log(
//         `i:${i}, ${arrow}, cost:${fnz(cost)},  w:${fnz(newWeight)}, dw:${fnz(
//           deltaWeight
//         )}, v:${fnz(neuron.output)}, gv:${fnz(prediction)}`
//       );
//       if (Math.abs(cost) < this.error) {
//         break;
//       }

//       weight = newWeight;
//       // value = newValue;
//     }

//     return { weight, steps: i, cost };
//   };

//   public testNeuronReal = (): LearningResult => {
//     const neuron = new Neuron();
//     neuron.debug = this.debug;
//     //neuron.init(1, 1);

//     // console.log(`0.12356 :${fnz(0.12356)}`);
//     //  console.log(`1.00165 :${fnz(1.00165)}`);
//     //   console.log(`0.000123 :${fnz(0.000123)}`);
//     //  console.log(`0.0000017 :${fnz(0.0000017)}`);
//     //  console.log(`10.03001 :${fnz(10.03001)}`);

//     //  return;

//     //let value = 0;
//     this.target = SharedFunctions.activationFunction(this.target);
//     //this.output = Math.random()*100;
//     let weight = 2; //Math.random()*5.0;
//     this.log(
//       `init o:${this.target}, w:${weight}, ld:${this.learningDelta}, e:${this.error}`
//     );

//     let i = 0;
//     let cost = 0;
//     for (i = 0; i < this.maxSteps; i++) {
//       // forward propagation
//       const prediction = neuron.prediction(weight, this.input);

//       // error find
//       cost = neuron.cost(this.target);

//       // new weight
//       //const dy = 1.0;
//       const dy = SharedFunctions.activationFunctionPrime(prediction); // why ?????
//       //const pD = this.ldelta * cost;
//       //const mDelta = Math.abs(pD) > this.ldelta ? pD : cost;

//       //const pD2 = mDelta * dy;
//       //const pDelta = Math.abs(pD2) > dy ? pD2 : mDelta;

//       const ccost = Math.abs(cost) > 1 ? cost : Math.sign(cost);

//       //const newWeight = weight + pDelta; // this.ldelta * delta * dy * this.input;
//       //const newWeight = weight + this.ldelta * cost * dy * this.input;
//       const newWeight = weight + this.learningDelta * ccost;
//       const arrow = cost > 0 ? 'i' : 'v';
//       const deltaWeight = weight - newWeight;

//       this.log(
//         `i:${i}, ${arrow}, cost:${fnz(cost)},  w:${fnz(newWeight)}, dw:${fnz(
//           deltaWeight
//         )}, v:${fnz(neuron.output)}, gv:${fnz(prediction)}, dy:${fnz(dy)}`
//       );
//       if (Math.abs(cost) < this.error) {
//         break;
//       }

//       weight = newWeight;
//       // value = newValue;
//     }
//     return { weight, steps: i, cost };
//   };

//   public testNeuronUnbounded = (): LearningResult => {
//     //let value = 0;
//     this.target = SharedFunctions.activationFunction(this.target);
//     //this.output = Math.random()*100;
//     let weight = 2; //Math.random()*5.0;
//     this.log(
//       `init o:${this.target}, w:${weight}, ld:${this.learningDelta}, e:${this.error}`
//     );

//     let i = 0;
//     let cost = 0;
//     for (i = 0; i < this.maxSteps; i++) {
//       // forward propagation
//       let value = this.input * weight;
//       //const newValue = value;
//       const newValue = SharedFunctions.activationFunction(value);

//       // error find
//       cost = this.target - newValue; // take sign only?

//       // new weight
//       //const dy = 1.0;
//       const dy = SharedFunctions.activationFunctionPrime(newValue);
//       const pD = this.learningDelta * cost;
//       const mDelta = Math.abs(pD) > this.learningDelta ? pD : cost;

//       const pD2 = mDelta * dy;
//       const pDelta = Math.abs(pD2) > dy ? pD2 : mDelta;

//       const newWeight = weight + pDelta; // this.ldelta * delta * dy * this.input;
//       const arrow = cost > 0 ? 'i' : 'v';
//       const deltaWeight = weight - newWeight;
//       this.log(
//         `i:${i}, ${arrow}, d:${cost.toFixed(3)},  w:${newWeight.toFixed(
//           3
//         )}, dw:${deltaWeight.toFixed(3)}, v:${value.toFixed(
//           3
//         )}, gv:${newValue.toFixed(3)}, dy:${dy.toFixed(3)}, pD:${pD.toFixed(
//           3
//         )}, pD2:${pD2.toFixed(3)}`
//       );
//       if (Math.abs(cost) < this.error) {
//         break;
//       }

//       weight = newWeight;
//       value = newValue;
//     }
//     return { weight, steps: i, cost };
//   };

//   public testNeuronGw = (): LearningResult => {
//     //let value = 0;
//     this.target = SharedFunctions.activationFunction(this.target);
//     //this.output = Math.random()*100;
//     let weight = 2; //Math.random()*5.0;
//     this.log(
//       `init o:${this.target}, w:${weight}, ld:${this.learningDelta}, e:${this.error}`
//     );

//     let i = 0;
//     let cost = 0;
//     for (i = 0; i < this.maxSteps; i++) {
//       // forward propagation
//       let value = this.input * weight;
//       //const newValue = value;
//       const newValue = SharedFunctions.activationFunction(value);

//       // error find
//       cost = this.target - newValue; // take sign only?

//       // new weight
//       //const dy = 1.0;
//       const dy = SharedFunctions.activationFunctionPrime(newValue);
//       const pD = this.learningDelta * cost;
//       const mDelta = Math.abs(pD) > this.learningDelta ? pD : cost;

//       const pD2 = mDelta * dy;
//       const pDelta = Math.abs(pD2) > dy ? pD2 : mDelta;

//       const newWeight = weight + pDelta; // this.ldelta * delta * dy * this.input;
//       const arrow = cost > 0 ? 'i' : 'v';
//       const deltaWeight = weight - newWeight;
//       this.log(
//         `i:${i}, ${arrow}, d:${cost.toFixed(3)},  w:${newWeight.toFixed(
//           3
//         )}, dw:${deltaWeight.toFixed(3)}, v:${value.toFixed(
//           3
//         )}, gv:${newValue.toFixed(3)}, dy:${dy.toFixed(3)}, pD:${pD.toFixed(
//           3
//         )}, pD2:${pD2.toFixed(3)}`
//       );
//       if (Math.abs(cost) < this.error) {
//         break;
//       }

//       weight = newWeight;
//       value = newValue;
//     }
//     return { weight, steps: i, cost };
//   };
//   /*
// public testNeuronOld = () => {
//     const n = new Neuron();
//     n.init();

//     for (let i = 0; i < 100; i++) {
//       n.value = this.input;

//       const deltaOut = n.value - this.output;
//       const df = Configuration.activationFunctionDf(n.value);
//       const newWeight =
//             n.weight + this.delta * deltaOut * this.input * df;

//       console.log(`nw correction i:${i}, d:${delta}, e:${this.error}, ow:${n.weight}, nw:${newWeight}`);
//       if (delta < this.error) {
//         break;
//       }

//       n.weight = newWeight;
//     }
//   };

//   private countValues = () =>
//   {

//   }*/

//   private log = (message?: unknown, optionalParams?: unknown[]): void => {
//     if (!this.debug) {
//       return;
//     }

//     if (optionalParams) {
//       console.log(message, ...optionalParams);
//       return;
//     }

//     console.log(message);
//   };
// }
