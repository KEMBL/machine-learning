/**
 * Useful string functions
 */
export class StringFunctions {
  public static fnz = (num: number, fixedVals = 3): string => {
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
}
