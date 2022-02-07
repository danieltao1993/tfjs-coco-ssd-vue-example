export const drawOnCanvas = (canvas, predictions) => {
  // get the context of canvas
  const ctx = canvas.getContext("2d");

  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  // If it's resolution does not match change it
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  // clear the canvas
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  console.log(
    " ctx.canvas.width, ctx.canvas.height: ",
    ctx.canvas.width,
    ctx.canvas.height
  );
  predictions.forEach((prediction) => {
    ctx.beginPath();
    ctx.rect(...prediction.bbox);
    ctx.lineWidth = 3;
    ctx.strokeStyle = "red";
    ctx.fillStyle = "red";
    ctx.stroke();
    ctx.shadowColor = "white";
    ctx.shadowBlur = 10;
    ctx.font = "24px Arial bold";
    ctx.fillText(
      `${(prediction.score * 100).toFixed(1)}% ${prediction.class}`,
      prediction.bbox[0],
      prediction.bbox[1] + 20
    );
  });
};


export const initContext = (canvas) => {
  const ctx = canvas.getContext("2d");

  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  // If it's resolution does not match change it
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  // clear the canvas
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  return ctx;
}

export const drawRects = (ctx, predictions) => {
  // get the context of canvas
  // console.log(
  //   " ctx.canvas.width, ctx.canvas.height: ",
  //   ctx.canvas.width,
  //   ctx.canvas.height
  // );
  predictions.forEach((prediction) => {
    ctx.beginPath();
    ctx.rect(...prediction.bbox);
    ctx.lineWidth = 3;
    ctx.strokeStyle = "red";
    ctx.fillStyle = "red";
    ctx.stroke();
    ctx.shadowColor = "white";
    ctx.shadowBlur = 10;
    ctx.font = "24px Arial bold";
    ctx.fillText(
      `${(prediction.score * 100).toFixed(1)}% ${prediction.class}`,
      prediction.bbox[0],
      prediction.bbox[1] + 20
    );
  });
};
