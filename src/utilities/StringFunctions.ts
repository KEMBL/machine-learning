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

    if (Math.abs(num) < 0.000000001) {
      return 'TOSMALL!';
    }

    try {
      return num.toFixed(
        1 - Math.floor(Math.log(Math.abs(num % 1)) / Math.log(10))
      );
    } catch (err) {
      console.log(`fnz error for num `, num, err);
      throw err;
    }
  };
}
