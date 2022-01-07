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
      <textarea v-model="JSON.stringify(predictions)" class="result"> </textarea>
    </div>
  </div>
</template>
<script>
// import img1 from "./assets/cars/1.jpeg";
// import img1 from "./assets/cars/1.jpeg";
// import img1 from "./assets/cars/1.jpeg";
// import img1 from "./assets/cars/1.jpeg";
// import img1 from "./assets/cars/1.jpeg";
import "@tensorflow/tfjs-backend-cpu";
import "@tensorflow/tfjs-backend-webgl";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import { drawOnCanvas } from "./utils/drawer";

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
    this.testCanvas();
    let t1 = new Date().valueOf();
    const model = await this.initModel();
    let t2 = new Date().valueOf();
    this.loadingConsumeTime = t2 - t1;
    this.model = model;
    this.predict();
  },
  methods: {
    onIndexChange(e) {
      this.imageIndex = e.target.value;
      setTimeout(() => {
        if (!this.model) return;
        this.predict();
      }, 500);
    },
    testCanvas() {
      const predictions = [
        {
          bbox: [
            0, 0, 500, 500,
            // 272.9414403438568, 161.1575186252594, 19.350886344909668,
            // 16.994327306747437,
          ],
          class: "test",
          score: 0,
        },
      ];
      drawOnCanvas(this.$refs.canvas, predictions);
    },
    async initModel() {
      this.isModelReady = false;
      const m = await cocoSsd.load();
      this.isModelReady = true;
      return m;
    },
    async predict() {
      // if (this.model) this.model.dispose();
      const image = this.$refs.img;
      let t1 = new Date().valueOf();
      this.predictions = await this.model.detect(image);
      let t2 = new Date().valueOf();
      this.predictConsumeTime = t2 - t1;
      drawOnCanvas(this.$refs.canvas, this.predictions);
      console.log("predictions: ", t2 - t1);
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
