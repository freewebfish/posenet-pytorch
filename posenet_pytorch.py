import torch
import cv2
import time
import argparse
import posenet

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=640)
parser.add_argument('--cam_height', type=int, default=480)
parser.add_argument('--mirror_flip', help='Optional. Do mirror flipping on the image frame.',
                    action='store_true')
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Optional. Use a video file instead of a live camera")
parser.add_argument('--max_poses', type=int, default=1, help="Max poses detected, an integer in [1, 30]")
args = parser.parse_args()

#python .\webcam_demo.py --cam_id 0 --cam_width 1280 --cam_height 960 --scale_factor 0.2 --model 75
#python .\webcam_demo.py --file D:\myData\VideoClips\Short-Exercise-720.mp4 --scale_factor 0.2 --model 75
def main():
    model = posenet.load_model(args.model)
    model = model.cuda()
    output_stride = model.output_stride

    mirror_flip = args.mirror_flip
    use_webcam = True
    if args.file is not None:
        cap = cv2.VideoCapture(args.file)
        use_webcam = False
    else:
        cap = cv2.VideoCapture(args.cam_id, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)

    maxPoses = args.max_poses
    if maxPoses < 1:
        print("cannot proceed -- max poses detected needs to be a positive integer!")
        return
    elif maxPoses > 30:
        print("would not proceed -- suggest max poses detected be no more than 30!")
        return

    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    print('\nPress [space] to pause/resume stream...\nPress [q] to quit stream...')
    start = time.time()
    frame_count = 0
    flag = True    
    while (cap.isOpened()):
        if cv2.waitKey(1) & 0xFF == ord(' '):
            flag = not(flag)

        if flag == True:            
            tik = time.time()
            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=args.scale_factor, output_stride=output_stride, mirror_flip=mirror_flip, use_webcam=use_webcam)
            if display_image is None:
                break

            with torch.no_grad():
                input_image = torch.Tensor(input_image).cuda()

                heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)

                pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                    heatmaps_result.squeeze(0),
                    offsets_result.squeeze(0),
                    displacement_fwd_result.squeeze(0),
                    displacement_bwd_result.squeeze(0),
                    output_stride=output_stride,
                    max_pose_detections=maxPoses,
                    min_pose_score=0.1)

            keypoint_coords *= output_scale
            tok = time.time()

            frame_count += 1
            # TODO this isn't particularly fast, use GL for drawing and display someday...
            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.1, min_part_score=0.1)
            cv2.putText(overlay_image, "FPS-overall: %.1f" % (frame_count / (time.time() - start)), (20, 20),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)                
            cv2.putText(overlay_image, "FPS-posenet: %.1f" % (1/(tok - tik)), (20, 40),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)                                
            cv2.imshow('posenet', overlay_image)
            #print('FPS: ', frame_count / (time.time() - start))                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            continue

    print("Average FPS: %.1f" %(frame_count / (time.time() - start)))
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()