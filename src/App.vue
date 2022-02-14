<template>
  <div id="app">
    <a-tabs :default-active-key="defaultActiveKey">
       <a-tab-pane key="NANODET" tab="NANODET">
        <NANODET />
      </a-tab-pane>
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
    </a-tabs>
  </div>
</template>

<script>

import Yolov5WithImage from "./Yolov5WithImage.vue";
import CocoSSDWithCam from "./CocoSSDWithCam.vue";
import ONNX from "./ONNX.vue";
import CocoImage from "./CocoImage.vue";
import NANODET from "./NANODET.vue";

export default {
  components: {
    Yolov5WithImage,
    CocoSSDWithCam,
    CocoImage,
    ONNX,
    NANODET,
  },
  name: "app",
  data() {
    return {
      defaultActiveKey: "Yolov5WithImage",
    };
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
