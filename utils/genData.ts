import * as tf from "@tensorflow/tfjs";
export const genBatch = async (
  sequences: tf.TensorBuffer<tf.Rank, "int32">,
  index: number[],
  batchSize: number,
  predictCount: number,
  maxLength: number
) => {
  const retX = tf.tensor(
    Array(batchSize * predictCount * (maxLength + 2)).fill(0),
    [batchSize, predictCount, maxLength + 2],
    "int32"
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

export default async function* (
  sequences: tf.Tensor,
  maxLength: number,
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
        predictCount,
        maxLength
      );
    }
  }
}
