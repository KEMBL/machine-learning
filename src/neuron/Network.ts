import { Neuron } from './Neuron';
import { Configuration } from './configuration/Configuration';
import { StringFunctions } from './utilities/StringFunctions';

const fnz = StringFunctions.fnz;

export class Network {
  /** input value */
  public input = 1.0;
  /** value which we want to achieve for given input */
  public target = 40.0;
  public error = 0.0001;
  public ldelta = 0.01;
  public maxSteps = 1000;
  // private neurons: Neuron[] = [];8

  //private i: number[];

  constructor() {
    //this.i = [0,1,2];
  }
  /*
  private init = (inNum:number, oNum) => {
    this.neurons = [];
    for(let i = 0; i < inNum; i++)
      {
        this.neurons.push(new Neuron);
        this.neurons[i].init();
      }
  }
*/

  public testNeuron = (type = 2): void => {
    switch (type) {
      case 1:
        this.testNeuronGw();
        break;
      case 2:
        this.testNeuronReal();
        break;
      default:
        this.testNeuronUnbounded();
        break;
    }
  };

  public testNeuronReal = (): void => {
    const neuron = new Neuron();
    //neuron.init(1, 1);

    // console.log(`0.12356 :${fnz(0.12356)}`);
    //  console.log(`1.00165 :${fnz(1.00165)}`);
    //   console.log(`0.000123 :${fnz(0.000123)}`);
    //  console.log(`0.0000017 :${fnz(0.0000017)}`);
    //  console.log(`10.03001 :${fnz(10.03001)}`);

    //  return;

    //let value = 0;
    this.target = Configuration.activationFunction(this.target);
    //this.output = Math.random()*100;
    let weight = 2; //Math.random()*5.0;
    console.log(`init o:${this.target}, w:${weight}, ld:${this.ldelta}, e:${this.error}`);

    for (let i = 0; i < this.maxSteps; i++) {
      // forward propagation
      const prediction = neuron.prediction(weight, this.input);

      // error find
      const cost = neuron.cost(this.target);

      // new weight
      //const dy = 1.0;
      const dy = Configuration.activationFunctionPrime(prediction);
      //const pD = this.ldelta * cost;
      //const mDelta = Math.abs(pD) > this.ldelta ? pD : cost;

      //const pD2 = mDelta * dy;
      //const pDelta = Math.abs(pD2) > dy ? pD2 : mDelta;

      const ccost = Math.abs(cost) > 1 ? cost : Math.sign(cost);

      //const newWeight = weight + pDelta; // this.ldelta * delta * dy * this.input;
      //const newWeight = weight + this.ldelta * cost * dy * this.input;
      const newWeight = weight + this.ldelta * ccost;
      const arrow = cost > 0 ? 'i' : 'v';
      const deltaWeight = weight - newWeight;

      console.log(`i:${i}, ${arrow}, cost:${fnz(cost)},  w:${fnz(newWeight)}, dw:${fnz(deltaWeight)}, v:${fnz(neuron.output)}, gv:${fnz(prediction)}, dy:${fnz(dy)}`);
      if (Math.abs(cost) < this.error) {
        break;
      }

      weight = newWeight;
      // value = newValue;
    }
  };

  public testNeuronUnbounded = (): void => {
    //let value = 0;
    this.target = Configuration.activationFunction(this.target);
    //this.output = Math.random()*100;
    let weight = 2; //Math.random()*5.0;
    console.log(`init o:${this.target}, w:${weight}, ld:${this.ldelta}, e:${this.error}`);

    for (let i = 0; i < this.maxSteps; i++) {
      // forward propagation
      let value = this.input * weight;
      //const newValue = value;
      const newValue = Configuration.activationFunction(value);

      // error find
      const delta = this.target - newValue; // take sign only?

      // new weight
      //const dy = 1.0;
      const dy = Configuration.activationFunctionPrime(newValue);
      const pD = this.ldelta * delta;
      const mDelta = Math.abs(pD) > this.ldelta ? pD : delta;

      const pD2 = mDelta * dy;
      const pDelta = Math.abs(pD2) > dy ? pD2 : mDelta;

      const newWeight = weight + pDelta; // this.ldelta * delta * dy * this.input;
      const arrow = delta > 0 ? 'i' : 'v';
      const deltaWeight = weight - newWeight;
      console.log(
        `i:${i}, ${arrow}, d:${delta.toFixed(3)},  w:${newWeight.toFixed(3)}, dw:${deltaWeight.toFixed(3)}, v:${value.toFixed(3)}, gv:${newValue.toFixed(3)}, dy:${dy.toFixed(
          3
        )}, pD:${pD.toFixed(3)}, pD2:${pD2.toFixed(3)}`
      );
      if (Math.abs(delta) < this.error) {
        break;
      }

      weight = newWeight;
      value = newValue;
    }
  };

  public testNeuronGw = (): void => {
    //let value = 0;
    this.target = Configuration.activationFunction(this.target);
    //this.output = Math.random()*100;
    let weight = 2; //Math.random()*5.0;
    console.log(`init o:${this.target}, w:${weight}, ld:${this.ldelta}, e:${this.error}`);

    for (let i = 0; i < this.maxSteps; i++) {
      // forward propagation
      let value = this.input * weight;
      //const newValue = value;
      const newValue = Configuration.activationFunction(value);

      // error find
      const delta = this.target - newValue; // take sign only?

      // new weight
      //const dy = 1.0;
      const dy = Configuration.activationFunctionPrime(newValue);
      const pD = this.ldelta * delta;
      const mDelta = Math.abs(pD) > this.ldelta ? pD : delta;

      const pD2 = mDelta * dy;
      const pDelta = Math.abs(pD2) > dy ? pD2 : mDelta;

      const newWeight = weight + pDelta; // this.ldelta * delta * dy * this.input;
      const arrow = delta > 0 ? 'i' : 'v';
      const deltaWeight = weight - newWeight;
      console.log(
        `i:${i}, ${arrow}, d:${delta.toFixed(3)},  w:${newWeight.toFixed(3)}, dw:${deltaWeight.toFixed(3)}, v:${value.toFixed(3)}, gv:${newValue.toFixed(3)}, dy:${dy.toFixed(
          3
        )}, pD:${pD.toFixed(3)}, pD2:${pD2.toFixed(3)}`
      );
      if (Math.abs(delta) < this.error) {
        break;
      }

      weight = newWeight;
      value = newValue;
    }
  };
  /*
public testNeuronOld = () => {
    const n = new Neuron();
    n.init();

    for (let i = 0; i < 100; i++) {
      n.value = this.input;

      const deltaOut = n.value - this.output;
      const df = Configuration.activationFunctionDf(n.value);
      const newWeight =
            n.weight + this.delta * deltaOut * this.input * df;

      console.log(`nw correction i:${i}, d:${delta}, e:${this.error}, ow:${n.weight}, nw:${newWeight}`);
      if (delta < this.error) {
        break;
      }

      n.weight = newWeight;
    }
  };

  private countValues = () =>
  {


  }*/
}
