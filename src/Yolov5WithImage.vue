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
// import tf from "@tensorflow/tfjs";
// import modelJson from "./models/best_web_model/model.json";
import { drawOnCanvas } from "./utils/drawer";

const tf = require("@tensorflow/tfjs");

function importAll(r) {
  return r.keys().map(r);
}
const images = importAll(
  require.context("./assets/cars/", false, /\.(png|jpe?g|svg)$/)
);

const names = [
  "person",
  "bicycle",
  "car",
  "motorcycle",
  "airplane",
  "bus",
  "train",
  "truck",
  "boat",
  "traffic light",
  "fire hydrant",
  "stop sign",
  "parking meter",
  "bench",
  "bird",
  "cat",
  "dog",
  "horse",
  "sheep",
  "cow",
  "elephant",
  "bear",
  "zebra",
  "giraffe",
  "backpack",
  "umbrella",
  "handbag",
  "tie",
  "suitcase",
  "frisbee",
  "skis",
  "snowboard",
  "sports ball",
  "kite",
  "baseball bat",
  "baseball glove",
  "skateboard",
  "surfboard",
  "tennis racket",
  "bottle",
  "wine glass",
  "cup",
  "fork",
  "knife",
  "spoon",
  "bowl",
  "banana",
  "apple",
  "sandwich",
  "orange",
  "broccoli",
  "carrot",
  "hot dog",
  "pizza",
  "donut",
  "cake",
  "chair",
  "couch",
  "potted plant",
  "bed",
  "dining table",
  "toilet",
  "tv",
  "laptop",
  "mouse",
  "remote",
  "keyboard",
  "cell phone",
  "microwave",
  "oven",
  "toaster",
  "sink",
  "refrigerator",
  "book",
  "clock",
  "vase",
  "scissors",
  "teddy bear",
  "hair drier",
  "toothbrush",
];
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
      // const weights = "https://raw.githubusercontent.com/raisa314/yolov5_object_detection/main/public/best_web_model/model.json";
      const weights = "https://zldrobit.github.io/web_model/model.json";
      // const weights = "/web_model/model.json";
      // const weights = "/best_web_model/model.json";
      this.isModelReady = false;
      let model = null;
      // const model = await loadGraphModel("http://127.0.0.1:8080/model.json");
      // const model = await tf.loadGraphModel("https://raw.githubusercontent.com/mdhasanali3/object-detection-with-yolov5-tfjs/master/public/web_model/model.json");
      // model = await tf.loadGraphModel(modelJson);
      model = await tf.loadGraphModel(weights);
      // model = await tf.loadModel(modelJson);
      console.log("model: ", model);
      this.isModelReady = true;
      return model;
    },
    getBoxFromData({
      valid_detections_data,
      boxes_data,
      scores_data,
      classes_data,
      canvas = { width: 300, height: 300 },
    }) {
      var i;
      const boxes = [];
      for (i = 0; i < valid_detections_data; ++i) {
        let [x1, y1, x2, y2] = boxes_data.slice(i * 4, (i + 1) * 4);
        x1 *= canvas.width;
        x2 *= canvas.width;
        y1 *= canvas.height;
        y2 *= canvas.height;
        const width = x2 - x1;
        const height = y2 - y1;
        const klass = names[classes_data[i]];
        const score = scores_data[i].toFixed(2);

        const box = {
          bbox: [x1, y1, width, height],
          class: klass,
          score: score,
        };
        boxes.push(box);
      }
      return boxes;
    },
    cropToCanvas(image, canvas, ctx) {
      const naturalWidth = image.naturalWidth;
      const naturalHeight = image.naturalHeight;

      // canvas.width = image.width;
      // canvas.height = image.height;

      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
      ctx.fillStyle = "#000000";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      const ratio = Math.min(
        canvas.width / image.naturalWidth,
        canvas.height / image.naturalHeight
      );
      const newWidth = Math.round(naturalWidth * ratio);
      const newHeight = Math.round(naturalHeight * ratio);
      ctx.drawImage(
        image,
        0,
        0,
        naturalWidth,
        naturalHeight,
        (canvas.width - newWidth) / 2,
        (canvas.height - newHeight) / 2,
        newWidth,
        newHeight
      );
    },
    // async predict() {
    //   // if (this.model) this.model.dispose();
    //   const image = this.$refs.img;
    //   let t1 = new Date().valueOf();
    //   this.predictions = await this.model.detect(image);
    //   let t2 = new Date().valueOf();
    //   this.predictConsumeTime = t2 - t1;
    //   drawOnCanvas(this.$refs.canvas, this.predictions);
    //   console.log("predictions: ", t2 - t1);
    // },
    async predict() {
      const image = this.$refs.img;
      const canvas = this.$refs.canvas;
      const ctx = canvas.getContext("2d");
      this.cropToCanvas(image, canvas, ctx);

      console.log("image: ", image);

      let t1 = new Date().valueOf();

      const input = tf.tidy(() => {
        return tf.image
          .resizeBilinear(tf.browser.fromPixels(image), [320, 320])
          .div(255.0)
          .expandDims(0);
      });
      const self = this;

      this.model.executeAsync(input).then((res) => {
        // Font options.
        const [boxes, scores, classes, valid_detections] = res;
        const boxes_data = boxes.dataSync();
        const scores_data = scores.dataSync();
        const classes_data = classes.dataSync();
        const valid_detections_data = valid_detections.dataSync()[0];
        const predictions = this.getBoxFromData({
          boxes_data,
          valid_detections_data,
          classes_data,
          scores_data,
          canvas: this.$refs.canvas,
        });
        console.log("predictions: ", predictions);
        drawOnCanvas(this.$refs.canvas, predictions);
        tf.dispose(res);
        let t2 = new Date().valueOf();
        self.predictConsumeTime = t2 - t1;
      });
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
  width: 320px;
  height: 320px;

  .sourceImage {
    position: absolute;
    width: 320px;
    height: 320px;
  }
  canvas {
    position: absolute;
    width: 320px;
    height: 320px;
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
