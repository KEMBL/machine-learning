import { Verbosity } from '../models';

export class Log {
  private static _globalVerbosity = Verbosity.None;
  private static _prefix?: string | number;

  public static set verbosity(value: Verbosity) {
    Log._globalVerbosity = value;
  }

  public static set prefix(value: string | number | undefined) {
    Log._prefix = value;
  }

  static log = (logLine: string, ...args: unknown[]): void => {
    Log.logLine(logLine, ...args);
  };

  static error = (
    logLine: string,
    sourceName?: string,
    ...args: unknown[]
  ): void => {
    Log.logVerbocity(Verbosity.Error, logLine, sourceName, ...args);
  };

  static warn = (
    logLine: string,
    sourceName?: string,
    ...args: unknown[]
  ): void => {
    Log.logVerbocity(Verbosity.Warning, logLine, sourceName, ...args);
  };

  static info = (
    logLine: string,
    sourceName?: string,
    ...args: unknown[]
  ): void => {
    Log.logVerbocity(Verbosity.Info, logLine, sourceName, ...args);
  };

  static debug = (
    logLine: string,
    sourceName?: string,
    ...args: unknown[]
  ): void => {
    Log.logVerbocity(Verbosity.Debug, logLine, sourceName, ...args);
  };

  private static logVerbocity = (
    verbosity: Verbosity,
    logLine: string,
    sourceName?: string,
    ...args: unknown[]
  ): void => {
    if (verbosity > Log._globalVerbosity) {
      return;
    }

    const prefix = Log._prefix !== undefined ? `${Log._prefix}> ` : '';
    Log.logLine(`${prefix}${sourceName}: ${logLine}`, ...args);
  };

  private static logLine = (logLine: string, ...args: unknown[]): void => {
    console.log(logLine, ...args);
  };
}
