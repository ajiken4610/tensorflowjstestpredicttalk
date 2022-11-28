<template lang="pug">
div
  button(@click="start") Start
</template>

<script setup lang="ts">
import data from "assets/daidoumei.txt?raw";
import * as tf from "@tensorflow/tfjs";

const start = async () => {
  // 前処理
  const lines = data.split("\n");
  const inputSequences: string[] = [];
  const charSet = new Set<string>();
  let maxLength = 0;
  const url_matcher = /.*https?:\/\/.*/;
  for (const line of lines) {
    if (line.match(url_matcher)) {
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
  const special_chars = ["<s>", "</s>", "<pad>"];
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
  const [trainSequence, testSequence] = tf.split(encodedSequences, [
    trainCount,
    inputSequences.length - trainCount,
  ]);
  console.log(maxLength, encodedSequences.shape[0], chars.length);
  //console.log(encodedSequences.arraySync());
};
</script>
