<template lang="pug">
div
  button(@click="init()") Start
  button(@click="buildModel()") BuildModel
  button(@click="fit()") Fit
</template>

<script setup lang="ts">
import data from "assets/daidoumei.txt?raw";
import * as tf from "@tensorflow/tfjs";

let trainSequence: tf.Tensor, testSequence: tf.Tensor;
let maxLength: number, charCount: number;
// let trashes: tf.Tensor[] = [];

const init = async () => {
  // 前処理
  const lines = data.split("\n");
  const inputSequences: string[] = [];
  const charSet = new Set<string>();
  maxLength = 0;
  const urlMatcher = /.*https?:\/\/.*/;
  for (const line of lines) {
    if (line.match(urlMatcher)) {
      continue;
    }
    const wakatied = line.split("");
    maxLength = Math.max(maxLength, wakatied.length);
    for (const char of wakatied) {
      charSet.add(char);
    }

    inputSequences.push(line);
  }
  const chars = Array.from(charSet).sort();
  const char2index: { [key: string]: number } = {};
  const index2char: { [key: number]: string } = {};
  for (var i = 0; i < chars.length; i++) {
    const c = chars[i];
    char2index[c] = i + 3;
    index2char[i + 3] = c;
  }

  const encodedSequences = tf.tensor(
    Array(inputSequences.length * (maxLength + 2)).fill(0),
    [inputSequences.length, maxLength + 2],
    "int32"
  );
  //const special_chars = ["<s>", "</s>", "<pad>"];
  const encodedBuffer = await encodedSequences.buffer();
  for (var i = 0; i < inputSequences.length; i++) {
    const line = inputSequences[i];
    encodedBuffer.set(1, i, 0);
    for (var j = 0; j < line.length; j++) {
      encodedBuffer.set(char2index[line.charAt(j)], i, j + 1);
    }
    encodedBuffer.set(2, i, line.length + 1);
  }

  const trainRatio = 0.7;
  const trainCount = Math.floor(inputSequences.length * trainRatio);
  [trainSequence, testSequence] = tf.split(encodedSequences, [
    trainCount,
    inputSequences.length - trainCount,
  ]);
  // const first16Array = Array(16 * maxLength).fill(0);
  // for (var i = 0; i < 16; i++) {
  //   first16Array[i * 16] = 1;
  //   first16Array[i * 16 + 1] = 2;
  // }
  // const first16 = tf.tensor(first16Array, [16, maxLength], "int32");
  // trainSequence = first16.concat(trainSequence);
  charCount = chars.length;
  console.log({
    maxLength,
    "encodedSequence.length": encodedSequences.shape[0],
    charCount: charCount,
  });
  encodedSequences.dispose();
};

const genBatch = async (
  sequences: tf.TensorBuffer<tf.Rank, "int32">,
  index: number[],
  batchSize: number,
  predictCount: number
) => {
  const retX = tf.tensor(
    Array(batchSize * predictCount * (maxLength + 2)).fill(0),
    [batchSize, predictCount, maxLength + 2],
    "float32"
  );
  const xBuf = await retX.buffer();
  const retT = tf.tensor(
    Array(batchSize * (maxLength + 2)),
    [batchSize, maxLength + 2],
    "int32"
  );
  const tBuf = await retT.buffer();
  for (var i = 0; i < batchSize; i++) {
    const cursor = index[i];
    for (var j = 0; j < predictCount; j++) {
      for (var k = 0; k < maxLength; k++) {
        xBuf.set(sequences.get(cursor - predictCount + j, k), i, j, k);
      }
    }
    for (var j = 0; j < maxLength; j++) {
      tBuf.set(sequences.get(cursor, j), i, j);
    }
  }
  // trashes.push(retX, retT);
  return { xs: retX, ys: retT };
};

const genData = async function* (
  sequences: tf.Tensor,
  batchSize = 128,
  predictCount = 16
) {
  const length = sequences.shape[0] - predictCount;
  const seqBuffer = (await sequences.buffer()) as tf.TensorBuffer<
    tf.Rank,
    "int32"
  >;
  const batchLength = Math.floor(length / batchSize);
  while (true) {
    const order = Array(length)
      .fill(null)
      .map((_, i) => i + predictCount);
    for (var i = 0; i < order.length; i++) {
      const index = Math.floor(Math.random() * order.length);
      [order[i], order[index]] = [order[index], order[i]];
    }
    for (var i = 0; i < batchLength; i++) {
      yield genBatch(
        seqBuffer,
        order.slice(i * batchSize, i * batchSize + batchSize),
        batchSize,
        predictCount
      );
    }
  }
};
let model: tf.Sequential;
const batchSize = 16;
const buildModel = () => {
  model = tf.sequential();

  // encoder
  model.add(tf.layers.flatten({ inputShape: [16, maxLength + 2] }));
  model.add(
    tf.layers.embedding({
      inputDim: charCount + 3,
      outputDim: 8,
      maskZero: true,
      inputLength: (maxLength + 2) * 16,
    })
  );
  model.add(tf.layers.reshape({ targetShape: [16, maxLength + 2, 8] }));
  model.add(
    tf.layers.timeDistributed({
      layer: tf.layers.bidirectional({
        layer: tf.layers.gru({ units: 8 /*32*/ }),
      }),
    })
  );
  model.add(
    tf.layers.bidirectional({ layer: tf.layers.gru({ units: 16 /* 64*/ }) })
  );
  model.add(tf.layers.reshape({ targetShape: [32 /* 128*/, 1] }));

  // decoder
  model.add(tf.layers.gru({ units: maxLength + 2, returnSequences: true }));
  model.add(tf.layers.permute({ dims: [2, 1] }));
  model.add(
    tf.layers.timeDistributed({
      layer: tf.layers.dense({ units: charCount + 3, activation: "softmax" }),
    })
  );

  const optimizer = new tf.AdagradOptimizer(0.001);
  model.compile({
    optimizer,
    loss: "sparseCategoricalCrossentropy",
    metrics: ["accuracy"],
  });
  model.summary();
};

const fit = async () => {
  const es = tf.callbacks.earlyStopping({
    monitor: "valLoss",
    patience: 5,
    verbose: 1,
  });
  const trainGenerator = genData(trainSequence, batchSize);
  const trainDataGenerator = tf.data
    .generator(async () => {
      const batch = (await trainGenerator.next()).value;
      const batchX = (await batch.xs.array()) as any[];
      const batchY = (await batch.xs.array()) as any[];
      return (function* () {
        for (var i = 0; i < batchSize; i++) {
          console.log("batch generated");
          yield { xs: batchX[i], ys: batchY[i] } as tf.TensorContainer;
        }
        return void 0;
      })();
    })
    .batch(1);
  const valGenerator = genData(testSequence, batchSize);
  const valDataGenerator = tf.data
    .generator(async () => {
      const batch = (await valGenerator.next()).value;
      const batchX = (await batch.xs.array()) as any[];
      const batchY = (await batch.xs.array()) as any[];
      return (function* () {
        for (var i = 0; i < batchSize; i++) {
          yield { xs: batchX[i], ys: batchY[i] } as tf.TensorContainer;
        }
        return void 0;
      })();
    })
    .batch(1);
  const history = await toRaw(model).fitDataset(trainDataGenerator, {
    batchesPerEpoch: 1,
    epochs: 1,
    verbose: 1,
    // validationData: tf.data.generator(
    //   () =>
    //     {}
    // ),
    // validationBatches: 16,
    callbacks: [
      // es,
      new (class extends tf.Callback {
        constructor() {
          super();
        }
        // async onBatchBegin(batch: number, logs: tf.Logs | undefined) {
        //   console.log("batchBigin:", batch);
        //   console.log(JSON.stringify(logs, null, 2));
        // }
        async onBatchEnd(batch: number, logs: tf.Logs | undefined) {
          console.log("batchEnd:", batch);
          console.log(JSON.stringify(logs, null, 2));
          // for (const trash of trashes) {
          //   trash.dispose();
          // }
        }
        // onYield(epoch: number, batch: number, logs: tf.Logs) {
        //   console.log("epoch:", epoch, "batch:", batch);
        // },
      })(),
    ],
  });
};
setInterval(() => {
  console.log(tf.memory());
}, 5000);
</script>
