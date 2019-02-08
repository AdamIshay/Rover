from RoverAPI import *
import argparse


if __name__ == '__main__':
    print('starting main')
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_name',
    	type=str,
    	help='name of the model file')

    parser.add_argument(
    	'--autonomous',
    	type=str,
    	default=True,
    	help='True for autonomous, False for human control. Default True.')

    parser.add_argument(
	   '--network',
       type=str,
       help='Number of the network that exists in the NetworkSwitch file.')

    parser.add_argument(
        '--driver',
        type=str,
        default='unknown_driver',
        help='The name of the person operating or running the rover.')

    parser.add_argument(
        '--rover',
        type=str,
        default='no_name',
        help='The name on the rover being used.')

    parser.add_argument(
        '--frames_per_second',
        type=int,
        default=30,
        help='The frame rate the rover will be operating at. Default 30')

    parser.add_argument(
        '--show_video_feed',
        type=bool,
        default=False,
        help="True to see rover's video feed, False to supress this feature.")

    parser.add_argument(
        '--save_training_data',
        type=str,
        default='y',
        help='Whether or not to save training data (y/n) if not autonomous.')

    parser.add_argument(
        '--ml_framework',
        type=str,
        default='tf',
        help='tf for TensorFlow model, pt for PyTorch model.')

    parser.add_argument(
        '--image_type',
        type=str,
        default='color',
        help='grayscale, color, or framestack. Default is color.')

    parser.add_argument(
        '--norm_method',
        type=str,
        default=None,
        help='Type instance_norm or channel_norm.')

    parser.add_argument(
        '--norm_vals',
        type=str,
        default='0,0,0',
        help='values to use in normalization.')

    parser.add_argument(
        '--num_outputs',
        type=int,
        default=4,
        help='The number of outputs for the network.')

    args = parser.parse_args()
    a = args.autonomous
    model_file_name = args.model_name
    network_name = args.network
    driver = args.driver
    rover = args.rover
    fps = args.frames_per_second
    view = args.show_video_feed
    data_save = args.save_training_data
    fw = args.ml_framework
    image_type = args.image_type
    normalization = args.norm_method
    norm_vals = [int(item) for item in args.norm_vals.split(',')]
    nout = args.num_outputs
    
# =============================================================================
#     if data_save in ['y', 'Y', 'yes', 'Yes'] and a is True:
#         data_save = False
# =============================================================================

# =============================================================================
#     model_file_name='/home/mpcr/Adam_python/Rover/RGB_model1.pt'
#     network_name='resnet34'
#     a=True
#     driver='unknown_driver'
#     rover='no_name'
#     fps=30
#     view=False
#     data_save='y'
#     fw='pt'
#     image_type='color'
#     normalization=None
#     norm_vals='0,0,0'
#     nout=4
# =============================================================================


    print('creating rover in main')

    rover = RoverRun(model_file_name,
                    network_name,
                    a,
                    driver,
                    rover,
                    fps,
                    view,
                    data_save,
                    fw,
                    image_type,
                    normalization,
                    norm_vals,
                    nout)
