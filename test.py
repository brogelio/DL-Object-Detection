from detecto import core
import os
import cv2
from opts import parse_opts

opt = parse_opts()
print(opt.video_path)
video_path = opt.video_path
video_path.replace(os.sep, '/')
model = core.Model.load('model_weights_v2.pth', ['Water', 'Soda', 'Juice'])
if opt.video_path == '0':
    cam = cv2.VideoCapture(0)
    while cam.isOpened():
        _, bgr_image = cam.read()
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        predictions = model.predict(rgb_image)

        labels, boxes, scores = predictions

        # Plot each box with its label and score
        for i in range(boxes.shape[0]):
            if scores[i] < 0.7:
                continue

            box = boxes[i]
            cv2.rectangle(bgr_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 3)
            if labels:
                cv2.putText(bgr_image, '{}: {}'.format(labels[i], round(scores[i].item(), 2)), (int(box[0]), int(box[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        cv2.imshow('Image', bgr_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cv2.destroyWindow('Image')
    cam.release()

else:
    cam = cv2.VideoCapture(opt.video_path)
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('output_file.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (frame_width, frame_height))
    while cam.isOpened():
        _, bgr_image = cam.read()
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        predictions = model.predict(rgb_image)

        labels, boxes, scores = predictions

        for i in range(boxes.shape[0]):
            if scores[i] < 0.7:
                continue

            box = boxes[i]
            cv2.rectangle(bgr_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 3)
            if labels:
                cv2.putText(bgr_image, '{}: {}'.format(labels[i], round(scores[i].item(), 2)),
                            (int(box[0]), int(box[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        out.write(bgr_image)

        if not _:
            break
        # If the 'q' or ESC key is pressed, break from the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cam.release()
    cv2.destroyAllWindows()
