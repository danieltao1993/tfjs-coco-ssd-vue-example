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
import "@tensorflow/tfjs-backend-cpu";
import "@tensorflow/tfjs-backend-webgl";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import { drawOnCanvas, initContext, drawRects } from "./utils/drawer";

import ndarray from "ndarray";
import ops from "ndarray-ops";
import { Tensor, InferenceSession } from "onnxjs";

// const ort = require("onnxruntime-web");

function importAll(r) {
  return r.keys().map(r);
}
const images = importAll(
  require.context("./assets/cars/", false, /\.(png|jpe?g|svg)$/)
);

// import * as images from "./assets/cars";

export default {
  data() {
    return {
      predictConsumeTime: 0,
      loadingConsumeTime: 0,
      predictions: [],
      images: images,
      imageIndex: 4,
      isModelReady: false,
    };
  },
  async mounted() {
    this.initCanvas();
    let t1 = new Date().valueOf();
    this.session = await this.initSession();
    let t2 = new Date().valueOf();
    this.loadingConsumeTime = t2 - t1;
    console.log("this.loadingConsumeTime: ", this.loadingConsumeTime);
    setTimeout(() => {
      this.predict();
    }, 300);
  },
  methods: {
    onIndexChange(e) {
      this.imageIndex = e.target.value;
      setTimeout(() => {
        if (!this.model) return;
        this.predict();
      }, 500);
    },
    initCanvas() {
      this.ctx = initContext(this.$refs.canvas);
      const canvas = this.$refs.canvas;
      this.ctx.drawImage(this.$refs.img, 0, 0, canvas.width, canvas.height);
    },

    async predict() {
      let t1 = new Date().valueOf();
      const tensor = this.preprocess(this.ctx);
      this.input_tensor = tensor;
      const boxes = await this.runModel(tensor);
      console.log("boxes: ", boxes);
      let t2 = new Date().valueOf();
      this.predictConsumeTime = t2 - t1;
      // drawRects(this.predictions)
    },
    async initSession() {
      let url = location.href + "onnx_model/" + "yolox_s.onnx";
      const session = await ort.InferenceSession.create(url,{
    executionProviders: ["webgl"],
  });
      return session;
    },

    preprocess(ctx) {
      const imageData = ctx.getImageData(
        0,
        0,
        ctx.canvas.width,
        ctx.canvas.height
      );
      const { data, width, height } = imageData;
      // data processing
      const dataTensor = ndarray(new Float32Array(data), [width, height, 4]);
      const dataProcessedTensor = ndarray(
        new Float32Array(width * height * 3),
        [1, 3, width, height]
      );

      ops.assign(
        dataProcessedTensor.pick(0, 0, null, null),
        dataTensor.pick(null, null, 0)
      );
      ops.assign(
        dataProcessedTensor.pick(0, 1, null, null),
        dataTensor.pick(null, null, 1)
      );
      ops.assign(
        dataProcessedTensor.pick(0, 2, null, null),
        dataTensor.pick(null, null, 2)
      );

      const tensor = new Tensor(
        new Float32Array(width * height * 3),
        "float32",
        [1, 3, width, height]
      );
      tensor.data.set(dataProcessedTensor.data);
      return tensor;
    },
    async runModel(inputTensor) {
      console.log("runModel: ", this.session, inputTensor);
      const outputData = await this.session.run([inputTensor]);
      const outputTensor = outputData.values().next().value;
      return this.postprocess(outputTensor, this.inferenceTime);
    },
    async postprocess(tensor, inferenceTime) {
      console.log("tensor: ", tensor);
      try {
        const originalOutput = new Tensor(
          tensor.data,
          "float32",
          [1, 125, 13, 13]
        );
        console.log("originalOutput: ", originalOutput);
        return originalOutput;
        // const outputTensor = yoloTransforms.transpose(
        //   originalOutput,
        //   [0, 2, 3, 1]
        // );

        // // postprocessing
        // const boxes = await yolo.postprocess(outputTensor, 20);
        // boxes.forEach((box) => {
        //   const { top, left, bottom, right, classProb, className } = box;

        //   this.drawRect(
        //     left,
        //     top,
        //     right - left,
        //     bottom - top,
        //     `${className} Confidence: ${Math.round(
        //       classProb * 100
        //     )}% Time: ${inferenceTime.toFixed(1)}ms`
        //   );
        // });
      } catch (e) {
        alert("Model is not valid!");
      }
    },
  },
};
</script>

<style lang="scss">
body {
  margin: 0;
  padding: 20px;
}
div {
  margin: 10px 0px;
}

.resultFrame {
  position: relative;
  width: 500px;
  height: 500px;

  .sourceImage {
    position: absolute;
    width: 500px;
    height: 500px;
  }
  canvas {
    position: absolute;
    width: 500px;
    height: 500px;
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
