<template>
  <div id="app">
    <a-tabs default-active-key="ONNX">
      <a-tab-pane key="ONNX" tab="ONNX">
        <ONNX />
      </a-tab-pane>
      <a-tab-pane key="CocoImage" tab="CocoImage">
        <CocoImage />
      </a-tab-pane>
      <a-tab-pane key="Yolov5WithImage" tab="Yolov5WithImage">
        <Yolov5WithImage />
      </a-tab-pane>
      <a-tab-pane key="CocoSSDWithCam" tab="CocoSSDWithCam">
        <CocoSSDWithCam />
      </a-tab-pane>
      <a-tab-pane key="WebNN" tab="WebNN">
        <WebNN />
      </a-tab-pane>
    </a-tabs>
  </div>
</template>

<script>
import "@tensorflow/tfjs-backend-cpu";
import "@tensorflow/tfjs-backend-webgl";
import Yolov5WithImage from "./Yolov5WithImage.vue";
import CocoSSDWithCam from "./CocoSSDWithCam.vue";
import ONNX from "./ONNX.vue";
import CocoImage from "./CocoImage.vue";
import WebNN from "./webnn/WebNN.vue";

export default {
  components: { Yolov5WithImage, CocoSSDWithCam, CocoImage,ONNX, WebNN },
  name: "app",
  data() {
    return {};
  },
  methods: {},
  mounted() {},
};
</script>

<style lang="scss">
body {
  margin: 0;
}
</style>
