<template>
  <div>
    <img class="cam" :src="images[0]" />
  </div>
</template>
<script>
import img1 from "./assets/1.jpeg";

function getBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve(reader.result);
    reader.onerror = (error) => reject(error);
  });
}

export default {
  data() {
    return {
      images: [img1],
    };
  },
  mounted() {
    this.images = [img1];
  },
  methods: {
    async initModal() {},
  },
};
</script>
<style>
.cam {
  width: 300px;
  height: 300px;
}
</style>
