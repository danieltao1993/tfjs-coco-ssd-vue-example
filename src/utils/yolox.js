import resizeImageData from "resize-image-data";
import nj from "numjs";
import ndarray from "ndarray";

export function preproc(img, input_size, swap = [2, 0, 1]) {
  let padded_img = nj.ones(input_size, "uint8").multiply(114);

  let r = Math.min(input_size[0] / img.shape[0], input_size[1] / img.shape[1]);
  const resized_img = resizeImageData(
    {
        data:img,
    },
    mg.shape[1] * r,
    img.shape[0] * r,
    "bilinear-interpolation"
  );
  
  padded_img = ndarray.transpose(padded_img, swap);
  return [padded_img, r];
}



// input_shape = tuple(map(int, args.input_shape.split(',')))
//     origin_img = cv2.imread(args.image_path)
//     img, ratio = preprocess(origin_img, input_shape)

//     session = onnxruntime.InferenceSession(args.model)

//     ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
//     output = session.run(None, ort_inputs)
//     predictions = demo_postprocess(output[0], input_shape, p6=args.with_p6)[0]

//     boxes = predictions[:, :4]
//     scores = predictions[:, 4:5] * predictions[:, 5:]

//     boxes_xyxy = np.ones_like(boxes)
//     boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
//     boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
//     boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
//     boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
//     boxes_xyxy /= ratio
//     dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
//     if dets is not None:
//         final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
//         origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
//                          conf=args.score_thr, class_names=COCO_CLASSES)

//     mkdir(args.output_dir)
//     output_path = os.path.join(args.output_dir, os.path.basename(args.image_path))
//     cv2.imwrite(output_path, origin_img)



// def demo_postprocess(outputs, img_size, p6=False):

//     grids = []
//     expanded_strides = []

//     if not p6:
//         strides = [8, 16, 32]
//     else:
//         strides = [8, 16, 32, 64]

//     hsizes = [img_size[0] // stride for stride in strides]
//     wsizes = [img_size[1] // stride for stride in strides]

//     for hsize, wsize, stride in zip(hsizes, wsizes, strides):
//         xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
//         grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
//         grids.append(grid)
//         shape = grid.shape[:2]
//         expanded_strides.append(np.full((*shape, 1), stride))

//     grids = np.concatenate(grids, 1)
//     expanded_strides = np.concatenate(expanded_strides, 1)
//     outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
//     outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

//     return outputs