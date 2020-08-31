import { Network } from '..';

/**
 * Useful string functions
 */
export class StringFunctions {
  public static fnz = (num: number, fixedVals = 4): string => {
    if (num === 0) {
      return num.toString();
    }

    if (Math.abs(num) >= 1) {
      return num.toFixed(fixedVals);
    }

    return num.toFixed(
      1 - Math.floor(Math.log(Math.abs(num % 1)) / Math.log(10))
    );
  };

  public static log = (logLine: string, ...args: unknown[]): void => {
    console.log(`${Network.currentStep}> ${logLine}`, ...args);
  };
}
