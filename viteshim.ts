declare module "*?raw" {
  const a = "";
  export = a;
}
declare module "*?worker" {
  class CustomizedWorker extends Worker {
    constructor();
  }
  export = CustomizedWorker;
}
