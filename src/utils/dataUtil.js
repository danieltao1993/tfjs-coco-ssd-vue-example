import * as tf from "@tensorflow/tfjs";

export async function prepareImageInput(image, [width, height]) {
  const buffer = tf.browser.fromPixels(image);
  const resize = tf.image.resizeBilinear(buffer, [width, height]);
  const cast = resize.cast("float32");
  const normalize = cast.div(255);
  const expand = normalize.expandDims(0);
  const transpose = expand.transpose([0, 3, 1, 2]);
  transpose.shape = transpose.shape.map(n=>Number(n))
  transpose.strides = transpose.strides.map(n=>Number(n))
  return transpose;
  // return tf.tidy(() => {
  //   let imageTensor = tf.browser.fromPixels(image, /* numChannels= */ 3);
  //   // Resize the query image according to the model input shape.
  //   imageTensor = tf.image.resizeBilinear(imageTensor, size[0], size[1], false);
  //   // Map to the correct input shape, range and type. The models expect float
  //   // inputs in the range [0, 1].
  //   imageTensor = imageTensor.toFloat().div(255).expandDims(0);
  //   return imageTensor;
  // });
}

export function imageDataToTensor(data, dims) {
  // 1. filter out alpha
  // 2. transpose from [224, 224, 3] -> [3, 224, 224]
  const [R, G, B] = [[], [], []];
  for (let i = 0; i < data.length; i += 4) {
    R.push(data[i]);
    G.push(data[i + 1]);
    B.push(data[i + 2]);
    // here we skip data[i + 3] because it's the alpha channel
  }
  const transposedData = R.concat(G).concat(B);

  // convert to float32
  let i,
    l = transposedData.length; // length, we need this for the loop
  const float32Data = new Float32Array(MAX_LENGTH); // create the Float32Array for output
  for (i = 0; i < l; i++) {
    float32Data[i] = transposedData[i] / MAX_SIGNED_VALUE; // convert to float
  }

  // return ort.Tensor
  const inputTensor = new ort.Tensor("float32", float32Data, dims);
  return inputTensor;
}

export async function loadImage(fileName, inputSize) {
  const data = fs.readFileSync(fileName);
  const obj = tf.tidy(() => {
    const buffer = tf.node.decodeImage(data);
    const resize = tf.image.resizeBilinear(buffer, [inputSize, inputSize]);
    const cast = resize.cast("float32");
    const normalize = cast.div(255);
    const expand = normalize.expandDims(0);
    const transpose = expand.transpose([0, 3, 1, 2]);
    const tensor = transpose;
    const img = {
      fileName,
      tensor,
      inputShape: [buffer.shape[1], buffer.shape[0]],
      outputShape: tensor.shape,
      size: buffer.size,
    };
    return img;
  });
  return obj;
}

/**
 * Read an image file as a TensorFlow.js tensor.
 *
 * Image resizing is performed with tf.image.resizeBilinear.
 *
 * @param {string} filePath Path to the input image file.
 * @param {number} height Desired height of the output image tensor, in pixels.
 * @param {number} width Desired width of the output image tensor, in pixels.
 * @return {tf.Tensor4D} The read float32-type tf.Tensor of shape
 *   `[1, height, width, 3]`
 */
export async function readImageTensorFromFile(filePath, height, width) {
  return new Promise((resolve, reject) => {
    jimp.read(filePath, (err, image) => {
      if (err) {
        reject(err);
      } else {
        const h = image.bitmap.height;
        const w = image.bitmap.width;
        const buffer = tf.buffer([1, h, w, 3], "float32");
        image.scan(0, 0, w, h, function (x, y, index) {
          buffer.set(image.bitmap.data[index], 0, y, x, 0);
          buffer.set(image.bitmap.data[index + 1], 0, y, x, 1);
          buffer.set(image.bitmap.data[index + 2], 0, y, x, 2);
        });
        resolve(
          tf.tidy(() =>
            tf.image.resizeBilinear(buffer.toTensor(), [height, width])
          )
        );
      }
    });
  });
}

export default {
  loadImage,
  readImageTensorFromFile,
  prepareImageInput,
  imageDataToTensor,
};
