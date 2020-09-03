export interface LayerConfig {
  neurons: number;
  activationType?: ActivationType;
}

export type ActivationType = 'ReLU' | 'LeakyReLU' | 'Sigmoid' | 'Identity';

export type CostType = 'Squared' | 'Default';
