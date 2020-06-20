import torch
import cv2
import time
import argparse
import pandas as pd
import datetime
import os
import posenet
import logging
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
args = parser.parse_args()


def main():
    now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')
    csvname = './output/keypoint' + now + '.csv'
    filename = './output/keypoint' + now + '.txt'
    with open(filename, 'w') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        model = posenet.load_model(args.model)
        model = model.cuda()
        output_stride = model.output_stride

        cap = cv2.VideoCapture(args.cam_id)
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)

        start = time.time()
        frame_count = 0
        while True:
            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=args.scale_factor, output_stride=output_stride)

            with torch.no_grad():
                input_image = torch.Tensor(input_image).cuda()

                heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)

                pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                    heatmaps_result.squeeze(0),
                    offsets_result.squeeze(0),
                    displacement_fwd_result.squeeze(0),
                    displacement_bwd_result.squeeze(0),
                    output_stride=output_stride,
                    max_pose_detections=10,
                    min_pose_score=0.15)

            keypoint_coords *= output_scale

            # TODO this isn't particularly fast, use GL for drawing and display someday...
            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)

            cv2.imshow('posenet', overlay_image)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            #print coordindates to screen, can also to csv file for later detection
            #dn = []
            for pi in range(len(pose_scores)):
                if pose_scores[pi] == 0.:
                    break
                print('Pose #%d, score = %f' % (pi, pose_scores[pi]))
                for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                    print('Keypoint %s, score = %f, coord = %s' % (posenet.PART_NAMES[ki], s, c))
                    #df1 = pd.DataFrame({'Pose': pi,
                    #                    'pose_scores': pose_scores[pi],
                    #                    'keypoint': posenet.PART_NAMES[ki],
                    #                    'score': s,
                    #                    'coord': c,
                    #                    }).set_index('Pose')
                    #dn.append(df1)

    print('Average FPS: ', frame_count / (time.time() - start))
    #dn.to_csv(csvname,index=True)

if __name__ == "__main__":
    main()