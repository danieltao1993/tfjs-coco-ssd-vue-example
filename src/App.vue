<template>
  <div id="app">
    <a-tabs default-active-key="Yolov5WithImage">
      <a-tab-pane key="Yolov5WithImage" tab="Yolov5WithImage">
        <Yolov5WithImage />
      </a-tab-pane>
      <a-tab-pane key="CocoSSDWithCam" tab="CocoSSDWithCam">
        <CocoSSDWithCam />
      </a-tab-pane>
    </a-tabs>
  </div>
</template>

<script>
import "@tensorflow/tfjs-backend-cpu";
import "@tensorflow/tfjs-backend-webgl";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import Yolov5WithImage from "./Yolov5WithImage.vue";
import CocoSSDWithCam from "./CocoSSDWithCam.vue";

export default {
  components: { Yolov5WithImage, CocoSSDWithCam },
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
