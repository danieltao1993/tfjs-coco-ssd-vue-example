<template>
  <div>
    <img class="cam" :src="images[0]" />
  </div>
</template>
<script>
import { TinyYoloV2Nchw } from "./tiny_yolov2_nchw.js";
import { TinyYoloV2Nhwc } from "./tiny_yolov2_nhwc.js";
import { SsdMobilenetV1Nchw } from "./ssd_mobilenetv1_nchw.js";
import { SsdMobilenetV1Nhwc } from "./ssd_mobilenetv1_nhwc.js";
import * as utils from "../common/utils.js";
// import * as Yolo2Decoder from "./libs/yolo2Decoder.js";
// import * as SsdDecoder from "./libs/ssdDecoder.js";

import img1 from "@/assets/test.jpg";

function constructNetObject(type) {
  const netObject = {
    tinyyolov2nchw: new TinyYoloV2Nchw(),
    tinyyolov2nhwc: new TinyYoloV2Nhwc(),
    ssdmobilenetv1nchw: new SsdMobilenetV1Nchw(),
    ssdmobilenetv1nhwc: new SsdMobilenetV1Nhwc(),
  };

  return netObject[type];
}

function getBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve(reader.result);
    reader.onerror = (error) => reject(error);
  });
}
async function fetchLabels(url) {
  // const response = await fetch(url);
  // const data = await response.text();
  // return data.split("\n");
  return ["bicycle", "bus", "car"];
}

export default {
  data() {
    return {
      images: [img1],
    };
  },
  mounted() {
    this.images = [img1];
    this.initModel();
  },
  methods: {
    async initModel() {
      let outputs = null;
      let devicePreference = "gpu";
      let modelName = "tinyyolov2";
      let layout = "nchw";
      let instanceType = modelName + layout;
      let netInstance = constructNetObject(instanceType);
      let inputOptions = netInstance.inputOptions;
      // let labels = await fetchLabels(inputOptions.labelUrl);
      let labels = ["bicycle", "bus", "car"];
      try {
        const outputOperand = await netInstance.load(devicePreference);
        console.log("outputOperand: ", outputOperand);
        netInstance.build(outputOperand);
      } catch (error) {
        debugger;
      }

      const inputBuffer = utils.getInputTensor(img1, inputOptions);
      console.log("inputBuffer: ", inputBuffer);
      const computeTimeArray = [];
      let medianComputeTime;

      if (modelName === "tinyyolov2") {
        outputs = {
          output: new Float32Array(
            utils.sizeOfShape(netInstance.outputDimensions)
          ),
        };
      } else {
        outputs = {
          boxes: new Float32Array(utils.sizeOfShape([1, 1917, 1, 4])),
          scores: new Float32Array(utils.sizeOfShape([1, 1917, 91])),
        };
      }
      if (numRuns > 1) {
        // Do warm up
        netInstance.compute(inputBuffer, outputs);
      }
      for (let i = 0; i < numRuns; i++) {
        start = performance.now();
        netInstance.compute(inputBuffer, outputs);
        computeTime = (performance.now() - start).toFixed(2);
        console.log(`  compute time ${i + 1}: ${computeTime} ms`);
        computeTimeArray.push(Number(computeTime));
      }
      if (numRuns > 1) {
        medianComputeTime = utils.getMedianValue(computeTimeArray);
        medianComputeTime = medianComputeTime.toFixed(2);
        console.log(`  median compute time: ${medianComputeTime} ms`);
      }
      console.log("output: ", outputs);

      // await drawOutput(imgElement, outputs, labels);
    },
  },
};
</script>
<style>
.cam {
  width: 1000px;
  height: auto;
}
</style>
