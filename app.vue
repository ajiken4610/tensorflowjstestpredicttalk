<template lang="pug">
div
  //- button(@click="init()") Start
  //- button(@click="buildModel()") BuildModel
  //- button(@click="fit()") Fit
</template>

<script setup lang="ts">
import data from "assets/daidoumei.txt?raw";
// import FitWorker from "assets/fitWorker?worker";
import * as tf from "@tensorflow/tfjs";
import word2index from "assets/word2index.json";
import index2word from "assets/index2word.json";
import lines from "assets/lines.json";

const model = await tf.loadLayersModel("/tp/model.json");
model.summary();

let maxLength: number = 40,
  charCount: number;
// 前処理
// const lines = data.split("\n");
// const inputSequences: string[] = [];
// const urlMatcher = /.*https?:\/\/.*/;
// for (const line of lines) {
//   if (line.match(urlMatcher)) {
//     continue;
//   }
//   inputSequences.push(line);
// }

// console.log(index2word);
// console.log(lines);
const encodedSequences = tf.tensor(lines, undefined, "int32");

// const encodedBuffer = await encodedSequences.buffer();
// for (var i = 0; i < inputSequences.length; i++) {
//   const line = inputSequences[i];
//   encodedBuffer.set(1, i, 0);
//   for (var j = 0; j < line.length; j++) {
//     encodedBuffer.set(word2index[line[j]], i, j + 1);
//   }
//   encodedBuffer.set(2, i, line.length + 1);
// }

const { xs, ys } = (await genData(encodedSequences, maxLength, 1).next())
  .value as {
  xs: tf.Tensor<tf.Rank>;
  ys: tf.Tensor<tf.Rank>;
};

const x = (xs.arraySync() as number[][][])[0];
const y = (ys.arraySync() as number[][])[0];

const p = model.predict(xs, { verbose: true }) as tf.Tensor;
const pArray = tf.tidy(() => {
  return tf
    .argMax(tf.tensor2d((p.arraySync() as any[])[0]))
    .arraySync() as number[];
});
p.dispose();
console.log(p.isDisposed);
// console.log(pArray);
console.log("last16=");
for (const s of x) {
  const print = [];
  for (const i of s) {
    print.push((index2word as any)[i]);
  }
  console.log(print.join(""));
}
console.log("correct:");
let print = [];
for (const i of y) {
  print.push((index2word as any)[i]);
}
console.log(print.join(""));
console.log("predict:");
print = [];
for (const i of pArray) {
  print.push((index2word as any)[i]);
}
console.log(print.join(""));
console.log(pArray);

// メモリたりないからCPUで。
// tf.setBackend("cpu");

// let trainSequence: tf.Tensor, testSequence: tf.Tensor;
// let maxLength: number, charCount: number;
// // let trashes: tf.Tensor[] = [];

// const init = async () => {
//   // 前処理
//   const lines = data.split("\n");
//   const inputSequences: string[] = [];
//   const charSet = new Set<string>();
//   maxLength = 0;
//   const urlMatcher = /.*https?:\/\/.*/;
//   for (const line of lines) {
//     if (line.match(urlMatcher)) {
//       continue;
//     }
//     const wakatied = line.split("");
//     maxLength = Math.max(maxLength, wakatied.length);
//     for (const char of wakatied) {
//       charSet.add(char);
//     }

//     inputSequences.push(line);
//   }
//   const chars = Array.from(charSet).sort();
//   const char2index: { [key: string]: number } = {};
//   const index2char: { [key: number]: string } = {};
//   for (var i = 0; i < chars.length; i++) {
//     const c = chars[i];
//     char2index[c] = i + 3;
//     index2char[i + 3] = c;
//   }

//   const encodedSequences = tf.tensor(
//     Array(inputSequences.length * (maxLength + 2)).fill(0),
//     [inputSequences.length, maxLength + 2],
//     "int32"
//   );
//   //const special_chars = ["<s>", "</s>", "<pad>"];
//   const encodedBuffer = await encodedSequences.buffer();
//   for (var i = 0; i < inputSequences.length; i++) {
//     const line = inputSequences[i];
//     encodedBuffer.set(1, i, 0);
//     for (var j = 0; j < line.length; j++) {
//       encodedBuffer.set(char2index[line.charAt(j)], i, j + 1);
//     }
//     encodedBuffer.set(2, i, line.length + 1);
//   }

//   const trainRatio = 0.7;
//   const trainCount = Math.floor(inputSequences.length * trainRatio);
//   [trainSequence, testSequence] = tf.split(encodedSequences, [
//     trainCount,
//     inputSequences.length - trainCount,
//   ]);
//   // const first16Array = Array(16 * maxLength).fill(0);
//   // for (var i = 0; i < 16; i++) {
//   //   first16Array[i * 16] = 1;
//   //   first16Array[i * 16 + 1] = 2;
//   // }
//   // const first16 = tf.tensor(first16Array, [16, maxLength], "int32");
//   // trainSequence = first16.concat(trainSequence);
//   charCount = chars.length;
//   console.log({
//     maxLength,
//     "encodedSequence.length": encodedSequences.shape[0],
//     charCount: charCount,
//   });
//   encodedSequences.dispose();
// };

// let model: tf.Sequential;
// const batchSize = 1;
// const buildModel = () => {
//   model = tf.sequential();

//   // encoder
//   model.add(tf.layers.flatten({ inputShape: [16, maxLength + 2] }));
//   model.add(
//     tf.layers.embedding({
//       inputDim: charCount + 3,
//       outputDim: 8,
//       maskZero: true,
//       inputLength: (maxLength + 2) * 16,
//     })
//   );
//   model.add(tf.layers.reshape({ targetShape: [16, maxLength + 2, 8] }));
//   model.add(
//     tf.layers.timeDistributed({
//       layer: tf.layers.bidirectional({
//         layer: tf.layers.gru({ units: 32 }),
//       }),
//     })
//   );
//   model.add(tf.layers.bidirectional({ layer: tf.layers.gru({ units: 64 }) }));
//   model.add(tf.layers.reshape({ targetShape: [128, 1] }));

//   // decoder
//   model.add(tf.layers.gru({ units: maxLength + 2, returnSequences: true }));
//   model.add(tf.layers.permute({ dims: [2, 1] }));
//   model.add(
//     tf.layers.timeDistributed({
//       layer: tf.layers.dense({ units: charCount + 3, activation: "softmax" }),
//     })
//   );

//   const optimizer = tf.train.adagrad(0.001);
//   model.compile({
//     optimizer,
//     loss: "sparseCategoricalCrossentropy",
//     metrics: ["accuracy"],
//   });
//   model.summary();
// };

// const fit = async () => {
//   // const worker = new FitWorker();

//   // worker.postMessage({
//   //   trainSequence,
//   //   maxLength,
//   //   batchSize,
//   //   testSequence,
//   //   model,
//   // });
//   const es = tf.callbacks.earlyStopping({
//     monitor: "valLoss",
//     patience: 5,
//     verbose: 1,
//   });
//   const trainGenerator = genData(trainSequence, maxLength, batchSize);
//   const trainDataGenerator = tf.data
//     .generator(async () => {
//       const batch = (await trainGenerator.next()).value as {
//         xs: tf.Tensor;
//         ys: tf.Tensor;
//       };
//       let batchX = (await batch.xs.array()) as any[];
//       let batchY = (await batch.ys.array()) as any[];
//       const batchXT: tf.Tensor[] = Array(batchSize);
//       const batchYT: tf.Tensor[] = Array(batchSize);
//       for (var i = 0; i < batchSize; i++) {
//         batchXT[i] = tf.tensor(batchX[i]);
//         batchYT[i] = tf.tensor(batchY[i]);
//       }
//       batchX = [];
//       batchY = [];
//       return (function* () {
//         for (var i = 0; i < batchSize; i++) {
//           yield { xs: batchXT[i], ys: batchYT[i] } as tf.TensorContainer;
//         }
//         return void 0;
//       })();
//     })
//     .batch(batchSize);

//   const valGenerator = genData(testSequence, 16);
//   const valDataGenerator = tf.data
//     .generator(async () => {
//       const batch = (await valGenerator.next()).value as {
//         xs: tf.Tensor;
//         ys: tf.Tensor;
//       };
//       const batchX = (await batch.xs.array()) as any[];
//       const batchY = (await batch.ys.array()) as any[];
//       batch.xs.dispose();
//       batch.ys.dispose();
//       return (function* () {
//         for (var i = 0; i < batchSize; i++) {
//           yield { xs: batchX[i], ys: batchY[i] } as tf.TensorContainer;
//         }
//         return void 0;
//       })();
//     })
//     .batch(16);
//   const history = await toRaw(model).fitDataset(trainDataGenerator, {
//     batchesPerEpoch: 1,
//     epochs: 1,
//     verbose: 1,
//     // validationData: tf.data.generator(
//     //   () =>
//     //     {}
//     // ),
//     // validationBatches: 16,
//     callbacks: [
//       // es,
//       new (class extends tf.Callback {
//         constructor() {
//           super();
//         }
//         // async onBatchBegin(batch: number, logs: tf.Logs | undefined) {
//         //   console.log("batchBigin:", batch);
//         //   console.log(JSON.stringify(logs, null, 2));
//         // }
//         async onBatchEnd(batch: number, logs: tf.Logs | undefined) {
//           console.log("batch:", batch);
//           console.log(logs);
//         }
//         async onEpochEnd(epoch: number, logs: tf.Logs | undefined) {
//           console.log("epoch:", epoch);
//           console.log(logs);
//         }
//         // onYield(epoch: number, batch: number, logs: tf.Logs) {
//         //   console.log("epoch:", epoch, "batch:", batch);
//         // },
//       })(),
//     ],
//     yieldEvery: 2500,
//   });
//   await model.save("indexeddb://model");
// };
// // setInterval(() => {
// //   console.log(tf.memory());
// // }, 5000);
</script>
