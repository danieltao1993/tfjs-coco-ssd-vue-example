<template>
  <div>
    <div class="resultFrame">
      <img ref="img" class="sourceImage" :src="images[imageIndex]" />
      <canvas ref="canvas"></canvas>
      <a-icon
        class="loading"
        v-if="!isModelReady"
        size="large"
        type="loading"
      />
    </div>
    <a-radio-group @change="onIndexChange" :default-value="imageIndex">
      <a-radio-button :key="img.name" v-for="(img, i) in images" :value="i">
        {{ `${img.slice(5).split(".")[0]}` }}
      </a-radio-button>
    </a-radio-group>
    <a-button @click="predict"> PREDICT </a-button>
    <div>
      <div>加载时间: {{ loadingConsumeTime }}(ms)</div>
      <div>预测时间: {{ predictConsumeTime }}(ms)</div>
      <textarea v-model="JSON.stringify(predictions)" class="result">
      </textarea>
    </div>
  </div>
</template>
<script>
import * as tf from "@tensorflow/tfjs";
// import "@tensorflow/tfjs-backend-cpu";
import "@tensorflow/tfjs-backend-webgl";
import { drawOnCanvas } from "./utils/drawer";
import dataUtil from "./utils/dataUtil";

function importAll(r) {
  return r.keys().map(r);
}
const images = importAll(
  require.context("./assets/cars/", false, /\.(png|jpe?g|svg)$/)
);

const modelOptions = {
  modelPath: "file://models/nanodet-t.json",
  minScore: 0.2, // low confidence, but still remove irrelevant
  iouThreshold: 0.4, // percentage when removing overlapped boxes
  maxResults: 20, // high number of results, but likely never reached
  scaleBox: 2.5, // increase box size
  activateScore: false, // use exponential function to active scores or use them as-is
};

export default {
  data() {
    return {
      predictConsumeTime: 0,
      loadingConsumeTime: 0,
      predictions: [],
      images: images,
      imageIndex: 5,
      isModelReady: false,
    };
  },
  async mounted() {
    // tf.ENV.set("WEBGL_PACK", false);
    // tf.env().setFlag("WEBGL_CPU_FORWARD", false);
    // tf.setBackend("cpu");

    this.testCanvas();
    let t1 = new Date().valueOf();
    const model = await this.initModel();
    let t2 = new Date().valueOf();
    this.loadingConsumeTime = t2 - t1;
    this.model = model;
    setTimeout(() => {
      this.predict();
    }, 0);
  },
  methods: {
    onIndexChange(e) {
      this.imageIndex = e.target.value;
      setTimeout(() => {
        if (!this.model) return;
        this.predict();
      }, 0);
    },
    testCanvas() {
      const predictions = [
        {
          bbox: [0, 0, 500, 500],
          class: "test",
          score: 0,
        },
      ];
      drawOnCanvas(this.$refs.canvas, predictions);
    },
    async initModel() {
      // const weights = "/nanodet_model/nanodet-t.json";
      // const weights = "/yolox_model/model.json";
      // const weights = "/best_web_model/model.json";
      const weights = "https://zldrobit.github.io/web_model/model.json";
      this.isModelReady = false;
      let model = null;
      model = await tf.loadGraphModel(weights);
      console.log("model: ", model);
      this.isModelReady = true;
      return model;
    },
    async predict() {
      const image = this.$refs.img;
      const _size = Object.values(this.model.modelSignature["inputs"])[0]
        .tensorShape.dim[2].size;
      const modelInputSize = ~~_size;
      console.log("image: ", image, modelInputSize);

      const tensor = await dataUtil.prepareImageInput(image, [
        modelInputSize,
        modelInputSize,
      ]);

      const res = this.model.predict(tensor);

      console.log("predict: ", res);

      const self = this;
      this.model.executeAsync(tensor).then(async (r) => {
        console.log("executeAsync: ", r);
        const boxes = await self.processResults(
          r,
          modelInputSize,
          tensor.shape
        );
        console.log("boxes: ", boxes);
      }, 10);
    },
    async processResults(res, inputSize, outputShape) {
      let results = [];
      let id = 0;
      for (const strideSize of [1, 2, 4]) {
        // try each stride size as it detects large/medium/small objects
        // find scores, boxes, classes
        // tf.tidy(() => {
        // wrap in tidy to automatically deallocate temp tensors
        const baseSize = strideSize * 13; // 13x13=169, 26x26=676, 52x52=2704
        // find boxes and scores output depending on stride
        const scoresT = res
          .find(
            (a) => a.shape[1] === baseSize ** 2 && a.shape[2] === labels?.length
          )
          ?.squeeze();
        const featuresT = res
          .find(
            (a) =>
              a.shape[1] === baseSize ** 2 &&
              (a.shape[2] === 32 || a.shape[2] === 44)
          )
          ?.squeeze();
        const boxesMax = featuresT.reshape([-1, 4, featuresT.shape[1] / 4]); // reshape [output] to [4, output / 4] where number is number of different features inside each stride
        const boxIdx = boxesMax.argMax(2).arraySync(); // what we need is indexes of features with highest scores, not values itself
        const scores = modelOptions.activateScore
          ? scoresT.exp(1).arraySync()
          : scoresT.arraySync(); // optionally use exponential scores or just as-is
        for (let i = 0; i < scoresT.shape[0]; i++) {
          // total strides (x * y matrix)
          for (let j = 0; j < scoresT.shape[1]; j++) {
            // one score for each class
            const score = scores[i][j] - (modelOptions.activateScore ? 1 : 0); // get score for current position
            // since original model is int64 based and tfjs casts it to int32 there are some overflows, most commonly around class 61 - so lets exclude those
            if (score > modelOptions.minScore && j !== 61) {
              const cx = (0.5 + Math.trunc(i % baseSize)) / baseSize; // center.x normalized to range 0..1
              const cy = (0.5 + Math.trunc(i / baseSize)) / baseSize; // center.y normalized to range 0..1
              const boxOffset = boxIdx[i].map(
                (a) => a * (baseSize / strideSize / inputSize)
              ); // just grab indexes of features with highest scores
              let boxRaw = [
                // results normalized to range 0..1
                cx - (modelOptions.scaleBox / strideSize) * boxOffset[0],
                cy - (modelOptions.scaleBox / strideSize) * boxOffset[1],
                cx + (modelOptions.scaleBox / strideSize) * boxOffset[2],
                cy + (modelOptions.scaleBox / strideSize) * boxOffset[3],
              ];
              boxRaw = boxRaw.map((a) => Math.max(0, Math.min(a, 1))); // fix out-of-bounds coords
              const box = [
                // results normalized to input image pixels
                boxRaw[0] * outputShape[0],
                boxRaw[1] * outputShape[1],
                boxRaw[2] * outputShape[0],
                boxRaw[3] * outputShape[1],
              ];
              const result = {
                id: id++,
                strideSize,
                score,
                class: j + 1,
                label: labels[j].label,
                center: [
                  Math.trunc(outputShape[0] * cx),
                  Math.trunc(outputShape[1] * cy),
                ],
                centerRaw: [cx, cy],
                box: box.map((a) => Math.trunc(a)),
                boxRaw,
              };
              results.push(result);
            }
          }
        }
        // });
      }

      // deallocate tensors
      res.forEach((t) => tf.dispose(t));

      // normally nms is run on raw results, but since boxes need to be calculated this way we skip calulcation of
      // unnecessary boxes and run nms only on good candidates (basically it just does IOU analysis as scores are already filtered)
      const nmsBoxes = results.map((a) => [
        a.box[1],
        a.box[0],
        a.box[3],
        a.box[2],
      ]); // switches boxes for nms from x,y to y,x
      const nmsScores = results.map((a) => a.score);
      const nms = await tf.image.nonMaxSuppressionAsync(
        nmsBoxes,
        nmsScores,
        modelOptions.maxResults,
        modelOptions.iouThreshold,
        modelOptions.minScore
      );
      const nmsIdx = nms.dataSync();
      tf.dispose(nms);
      // filter & sort results
      results = results
        .filter((a, idx) => nmsIdx.includes(idx))
        .sort((a, b) => b.score - a.score);
      return results;
    },
  },
};
</script>

<style lang="scss" scoped>
body {
  margin: 0;
  padding: 20px;
}
div {
  margin: 10px 0px;
}

.resultFrame {
  position: relative;
  width: 640px;
  height: 640px;

  .sourceImage {
    position: absolute;
    width: 640px;
    height: 640px;
  }
  canvas {
    position: absolute;
    width: 640px;
    height: 640px;
  }
  .loading {
    color: #f00;
    position: absolute;
    top: 200px;
    left: 200px;
    transform: scale(5);
  }
}
.result {
  width: 400px;
  height: auto;
  min-height: 200px;
}
</style>
