/**
 * Useful string functions
 */
class StringFunctions {
  public fnz = (num: number, fixedVals = 3): string => {
    if (Math.abs(num) >= 1) {
      return num.toFixed(fixedVals);
    }

    return num.toFixed(1 - Math.floor(Math.log(Math.abs(num % 1)) / Math.log(10)));
  };
}

const stringFunctions = new StringFunctions();
export { stringFunctions as StringFunctions };
